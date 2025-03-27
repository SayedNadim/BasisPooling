import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import copy

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a simple CNN with capacity for both MNIST and CIFAR
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Dictionary to store activation statistics for each layer
        self.activation_stats = {}
        self.register_hooks()
        
    def register_hooks(self):
        """Register forward hooks to collect activation statistics"""
        def get_activation_stats(name):
            def hook(module, input, output):
                self.activation_stats[name] = output.detach()
            return hook
        
        # Register hooks for convolutional layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.register_forward_hook(get_activation_stats(name))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_fisher_information(self, data_loader, num_samples=100, device='cuda'):
        """Calculate Fisher information for each parameter (approximates parameter importance)"""
        fisher_dict = {}
        
        # Initialize Fisher information for each parameter
        for name, param in self.named_parameters():
            fisher_dict[name] = torch.zeros_like(param.data)
        
        # Set model to evaluation mode
        self.eval()
        
        # Use cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        
        # Process a subset of data to estimate Fisher information
        sample_count = 0
        for inputs, targets in data_loader:
            if sample_count >= num_samples:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            sample_count += inputs.size(0)
            
            # Forward pass
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            self.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
        
        # Normalize by number of samples
        for name in fisher_dict:
            fisher_dict[name] /= sample_count
            
        return fisher_dict


# Load datasets
def load_mnist():
    transform = transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),  # Resize to match CIFAR dimensions
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

# Train a model on a dataset
def train_model(model, train_loader, epochs=5, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.3f}%')
                running_loss = 0.0
    
    return model

# Evaluate a model on a dataset
def evaluate_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Accuracy on test set: {accuracy:.3f}%')
    return accuracy

class ModelMerger:
    @staticmethod
    def layer_wise_merging(model_A, model_B, strategy_map=None, device='cuda'):
        """Apply different merging strategies to different layers"""
        if strategy_map is None:
            # Default strategy map: early layers averaged, later layers from model A
            strategy_map = {
                'features.0': 'average',  # First conv layer
                'features.3': 'average',  # Second conv layer
                'features.6': 'model_B',  # Third conv layer
                'classifier.0': 'model_B', # First FC layer
                'classifier.3': 'model_A'  # Final FC layer
            }
        
        merged_model = copy.deepcopy(model_A)
        merged_model.to(device)
        model_A.to(device)
        model_B.to(device)
        
        # Get state dictionaries
        state_A = model_A.state_dict()
        state_B = model_B.state_dict()
        merged_state = merged_model.state_dict()
        
        # Apply layer-wise merging strategies
        for name, param in merged_model.named_parameters():
            # Find the closest matching layer in the strategy map
            matching_layer = None
            for layer_name in strategy_map:
                if layer_name in name:
                    matching_layer = layer_name
                    break
            
            # Apply the appropriate strategy
            if matching_layer and matching_layer in strategy_map:
                strategy = strategy_map[matching_layer]
                
                if strategy == 'average':
                    merged_state[name] = (state_A[name] + state_B[name]) / 2.0
                elif strategy == 'model_A':
                    merged_state[name] = state_A[name]
                elif strategy == 'model_B':
                    merged_state[name] = state_B[name]
                # Add more strategies as needed
            else:
                # Default to averaging if no strategy specified
                merged_state[name] = (state_A[name] + state_B[name]) / 2.0
        
        # Load the merged state
        merged_model.load_state_dict(merged_state)
        return merged_model
    
    @staticmethod
    def fisher_weighted_merging(model_A, model_B, loader_A, loader_B, device='cuda'):
        """Merge weights using Fisher information for importance weighting"""
        merged_model = copy.deepcopy(model_A)
        merged_model.to(device)
        model_A.to(device)
        model_B.to(device)
        
        # Calculate Fisher information for both models
        print("Calculating Fisher information for model A...")
        fisher_A = model_A.get_fisher_information(loader_A, device=device)
        print("Calculating Fisher information for model B...")
        fisher_B = model_B.get_fisher_information(loader_B, device=device)
        
        # Merge parameters based on relative Fisher importance
        with torch.no_grad():
            for name, param in merged_model.named_parameters():
                # Get Fisher information for this parameter
                f_A = fisher_A.get(name, torch.zeros_like(param.data))
                f_B = fisher_B.get(name, torch.zeros_like(param.data))
                
                # Calculate weights based on relative importance
                weight_A = f_A / (f_A + f_B + 1e-10)  # Add small epsilon to avoid division by zero
                weight_B = 1 - weight_A
                
                # Apply weighted average
                param_A = model_A.state_dict()[name]
                param_B = model_B.state_dict()[name]
                param.data = weight_A * param_A + weight_B * param_B
        
        return merged_model
    

    @staticmethod
    def naive_average(model_A, model_B):
        """Simple averaging of weights"""
        merged_model = copy.deepcopy(model_A)
        
        with torch.no_grad():
            for (name_A, param_A), (name_B, param_B) in zip(model_A.named_parameters(), model_B.named_parameters()):
                # Check if shapes match
                if param_A.shape != param_B.shape:
                    print(f"Skipping parameter {name_A} due to shape mismatch: {param_A.shape} vs {param_B.shape}")
                    continue
                
                param_merged = merged_model.get_parameter(name_A)
                param_merged.data = (param_A.data + param_B.data) / 2.0
                
        return merged_model
    
    @staticmethod
    def similarity_based_gating(model_A, model_B, threshold=0.3):
        """Merge weights based on cosine similarity
        If similarity is high, average the weights
        If similarity is low, keep both contributions using a weighted sum
        """
        merged_model = copy.deepcopy(model_A)
        
        with torch.no_grad():
            for (name_A, param_A), (name_B, param_B) in zip(model_A.named_parameters(), model_B.named_parameters()):
                # Check if shapes match
                if param_A.shape != param_B.shape:
                    print(f"Skipping parameter {name_A} due to shape mismatch: {param_A.shape} vs {param_B.shape}")
                    continue
                
                param_merged = merged_model.get_parameter(name_A)
                
                # Reshape parameters for cosine similarity calculation
                flat_A = param_A.data.view(-1)
                flat_B = param_B.data.view(-1)
                
                # Calculate cosine similarity
                if flat_A.norm() > 0 and flat_B.norm() > 0:  # Avoid division by zero
                    similarity = torch.nn.functional.cosine_similarity(flat_A.unsqueeze(0), flat_B.unsqueeze(0)).item()
                else:
                    similarity = 0
                
                # Apply similarity-based merging strategy
                if similarity > threshold:
                    # High similarity: simple average
                    param_merged.data = (param_A.data + param_B.data) / 2.0
                else:
                    # Low similarity: weighted contribution based on parameter magnitudes
                    weight_A = param_A.data.norm() / (param_A.data.norm() + param_B.data.norm())
                    weight_B = 1 - weight_A
                    param_merged.data = weight_A * param_A.data + weight_B * param_B.data
        
        return merged_model
    
    @staticmethod
    def gated_softmax_merging(model_A, model_B, alpha=1.0, beta=1.0):
        """Merge weights using a softmax gating approach"""
        merged_model = copy.deepcopy(model_A)
        
        with torch.no_grad():
            for (name_A, param_A), (name_B, param_B) in zip(model_A.named_parameters(), model_B.named_parameters()):
                # Check if shapes match
                if param_A.shape != param_B.shape:
                    print(f"Skipping parameter {name_A} due to shape mismatch: {param_A.shape} vs {param_B.shape}")
                    continue
                
                param_merged = merged_model.get_parameter(name_A)
                
                # Calculate importance scores (using norm as a simple proxy for importance)
                score_A = alpha * param_A.data.norm()
                score_B = beta * param_B.data.norm()
                
                # Calculate softmax weights
                exp_A = torch.exp(score_A)
                exp_B = torch.exp(score_B)
                weight_A = exp_A / (exp_A + exp_B)
                weight_B = exp_B / (exp_A + exp_B)
                
                # Apply weighted average
                param_merged.data = weight_A * param_A.data + weight_B * param_B.data
        
        return merged_model
    
    @staticmethod
    def dropout_inspired_merging(model_A, model_B, dropout_prob=0.5):
        """Randomly select weights from either model A or B"""
        merged_model = copy.deepcopy(model_A)
        
        with torch.no_grad():
            for (name_A, param_A), (name_B, param_B) in zip(model_A.named_parameters(), model_B.named_parameters()):
                # Check if shapes match
                if param_A.shape != param_B.shape:
                    print(f"Skipping parameter {name_A} due to shape mismatch: {param_A.shape} vs {param_B.shape}")
                    continue
                
                param_merged = merged_model.get_parameter(name_A)
                
                # Create a random mask for selecting parameters
                mask = torch.rand_like(param_A.data) > dropout_prob
                
                # Apply mask to select parameters from each model
                param_merged.data = torch.where(mask, param_A.data, param_B.data)
        
        return merged_model

# Main experiment function
def run_experiment(device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Using device: {device}")
    
    # Load datasets
    mnist_train, mnist_test = load_mnist()
    cifar_train, cifar_test = load_cifar10()
    
    # Initialize models
    model_mnist = SimpleCNN(num_classes=10, input_channels=3)
    model_cifar = SimpleCNN(num_classes=10, input_channels=3)
    
    # Train models on respective datasets
    print("Training MNIST model...")
    model_mnist = train_model(model_mnist, mnist_train, epochs=5, device=device)
    
    print("Training CIFAR-10 model...")
    model_cifar = train_model(model_cifar, cifar_train, epochs=5, device=device)
    
    # Evaluate base models
    print("\nEvaluating base models:")
    print("MNIST model on MNIST:")
    mnist_acc_A = evaluate_model(model_mnist, mnist_test, device=device)
    print("CIFAR model on CIFAR:")
    cifar_acc_B = evaluate_model(model_cifar, cifar_test, device=device)
    
    # Create merged models using different strategies
    merging_strategies = {
        "Naive Average": ModelMerger.naive_average(model_mnist, model_cifar),
        "Similarity Gating": ModelMerger.similarity_based_gating(model_mnist, model_cifar),
        "Gated Softmax": ModelMerger.gated_softmax_merging(model_mnist, model_cifar),
        "Dropout Inspired": ModelMerger.dropout_inspired_merging(model_mnist, model_cifar),
        "Layer-wise Merging": ModelMerger.layer_wise_merging(model_mnist, model_cifar),
        "Fisher Weighted Merging": ModelMerger.fisher_weighted_merging(model_mnist, model_cifar, mnist_train, cifar_train)

    }
    
    # Evaluate merged models
    results = {
        "MNIST": {"Base Model": mnist_acc_A},
        "CIFAR": {"Base Model": cifar_acc_B}
    }
    
    print("\nEvaluating merged models:")
    for strategy_name, merged_model in merging_strategies.items():
        print(f"\n--- {strategy_name} ---")
        
        # Evaluate on MNIST
        print("Evaluating on MNIST:")
        mnist_acc = evaluate_model(merged_model, mnist_test, device=device)
        results["MNIST"][strategy_name] = mnist_acc
        
        # Adjust first layer for CIFAR evaluation (1 channel -> 3 channels)
        cifar_eval_model = copy.deepcopy(merged_model)
        
        # Evaluate on CIFAR
        print("Evaluating on CIFAR:")
        cifar_acc = evaluate_model(cifar_eval_model, cifar_test, device=device)
        results["CIFAR"][strategy_name] = cifar_acc
    
    # Plot results
    plot_results(results)
    
    return results

def plot_results(results):
    # Extract data for plotting
    strategies = list(results["MNIST"].keys())
    mnist_acc = [results["MNIST"][s] for s in strategies]
    cifar_acc = [results["CIFAR"][s] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, mnist_acc, width, label='MNIST')
    rects2 = ax.bar(x + width/2, cifar_acc, width, label='CIFAR-10')
    
    ax.set_title('Performance of Different Merging Strategies')
    ax.set_xlabel('Merging Strategy')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('merging_strategies_comparison.png')
    plt.show()

if __name__ == "__main__":
    results = run_experiment()
    print("\nExperiment completed!")