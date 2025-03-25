import torch
import torch.nn as nn
import torch.nn.functional as F


# Fast implementation with even fewer operations
class BasisPooling(nn.Module):
    """
    Ultra-fast implementation of Basis Pooling that approximates the selection
    mechanism with minimal computational overhead.
    """
    def __init__(self, kernel_size, stride=None, basis_dim=2, temperature=0.1):
        super(BasisPooling, self).__init__()
        
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
            
        self.basis_dim = basis_dim
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Calculate output dimensions
        out_height = (height - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width - self.kernel_size[1]) // self.stride[1] + 1
        
        # Extract patches
        patches = F.unfold(
            x, 
            kernel_size=self.kernel_size, 
            stride=self.stride
        ).view(batch_size, channels, -1, out_height*out_width)
        
        if self.basis_dim == 1:
            # If basis_dim is 1, just return max pooling for efficiency
            output = patches.max(dim=2)[0].view(batch_size, channels, out_height, out_width)
            return output
        
        outputs = []
        
        # First basis: max pooling (most important feature)
        max_values, _ = patches.max(dim=2)
        outputs.append(max_values)
        
        if self.basis_dim > 1:
            # Second basis: compute complement information
            # We use min pooling for the second basis as it often contains complementary info
            min_values, _ = patches.min(dim=2)
            outputs.append(min_values)
        
        # For additional bases, use statistical moments (mean, variance, etc.)
        if self.basis_dim > 2:
            # Third basis: mean values
            mean_values = patches.mean(dim=2)
            outputs.append(mean_values)
            
        if self.basis_dim > 3:
            # Fourth and beyond: higher statistical moments and other features
            for i in range(3, self.basis_dim):
                if i == 3:
                    # Variance
                    moment = ((patches - patches.mean(dim=2, keepdim=True))**2).mean(dim=2)
                elif i == 4:
                    # Median
                    moment, _ = patches.median(dim=2)
                else:
                    # Higher moments (approximated)
                    power = (i - 2) / 2 + 1
                    moment = ((patches - patches.mean(dim=2, keepdim=True))**power).mean(dim=2)
                
                outputs.append(moment)
        
        # Combine all basis vectors
        output = torch.stack(outputs, dim=2)
        output = output.view(batch_size, channels * self.basis_dim, out_height, out_width)
        
        return output



class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        
        # Use Basis Pooling instead of MaxPool
        self.pool1 = BasisPooling(kernel_size=2, basis_dim=4, temperature=0.1)
        
        # Note: After basis pooling, we have 16*2=32 channels
        self.conv2 = nn.Conv2d(16*4, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        
        # Another basis pooling layer
        self.pool2 = BasisPooling(kernel_size=2, basis_dim=4, temperature=0.1)
        
        # Fully connected layers
        self.fc = nn.Linear(32*4*8*8, num_classes)  # Assuming 32x32 input images
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


# Function to test and compare different pooling methods
def compare_pooling_methods():
    # Create a reproducible random input tensor
    torch.manual_seed(42)
    batch_size, channels, height, width = 2, 3, 8, 8
    x = torch.randn(batch_size, channels, height, width)
    
    # Create synthetic features for better visualization
    # Add some distinct patterns in different regions
    for b in range(batch_size):
        # Strong edge in upper left
        x[b, 0, 0:2, 0:2] = 2.0
        # Texture pattern in middle
        x[b, 1, 3:5, 3:5] = torch.tensor([[1.5, -1.5], [-1.5, 1.5]])
        # Gradient in bottom right
        x[b, 2, 6:8, 6:8] = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    
    # Define pooling layers
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    basis_pool = BasisPooling(kernel_size=2, basis_dim=2, temperature=0.1)
    
    # Forward passes
    max_output = max_pool(x)
    avg_output = avg_pool(x)
    basis_output = basis_pool(x)
    
    # Print input and shapes
    print("Input tensor:")
    print(x[0, :, :, :])  # First batch item
    print(f"Input shape: {x.shape}")
    
    print("\nMax Pool output:")
    print(max_output[0, :, :, :])
    print(f"Max Pool output shape: {max_output.shape}")
    
    print("\nAvg Pool output:")
    print(avg_output[0, :, :, :])
    print(f"Avg Pool output shape: {avg_output.shape}")
    
    print("\nBasis Pool output (first half of channels):")
    print(basis_output[0, :channels, :, :])
    print("\nBasis Pool output (second half of channels):")
    print(basis_output[0, channels:, :, :])
    print(f"Basis Pool output shape: {basis_output.shape}")
    
    # Check for information preservation
    # Try reconstructing the original pattern from the pooled outputs
    
    # Simplistic reconstruction (upsampling by repeating values)
    def simple_upsample(tensor):
        return tensor.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
    
    max_reconstruction = simple_upsample(max_output)
    avg_reconstruction = simple_upsample(avg_output)
    
    # For basis pooling, we'll use the average of the basis vectors
    basis_reconstruction = torch.zeros_like(x)
    for c in range(channels):
        # Average the basis vectors for each original channel
        basis_avg = (basis_output[:, c*2:(c+1)*2, :, :].mean(dim=1, keepdim=True))
        basis_reconstruction[:, c:c+1, :, :] = simple_upsample(basis_avg)
    
    # Calculate reconstruction error
    max_error = F.mse_loss(x, max_reconstruction).item()
    avg_error = F.mse_loss(x, avg_reconstruction).item()
    basis_error = F.mse_loss(x, basis_reconstruction).item()
    
    print("\nReconstruction Mean Squared Error:")
    print(f"Max Pool MSE: {max_error:.4f}")
    print(f"Avg Pool MSE: {avg_error:.4f}")
    print(f"Basis Pool MSE: {basis_error:.4f}")
    
    return {
        'input': x,
        'max_output': max_output,
        'avg_output': avg_output,
        'basis_output': basis_output,
        'max_reconstruction': max_reconstruction,
        'avg_reconstruction': avg_reconstruction,
        'basis_reconstruction': basis_reconstruction,
        'errors': {
            'max': max_error,
            'avg': avg_error,
            'basis': basis_error
        }
    }

# Function to test with MNIST or similar dataset
def compare_on_mnist(num_epochs=2):
    try:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        import time
        
        # Load a small subset of MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        subset_indices = torch.randperm(len(dataset))[:1000]  # Use 1000 samples for speed
        subset = torch.utils.data.Subset(dataset, subset_indices)
        
        loader = DataLoader(subset, batch_size=64, shuffle=True)
        
        # Define three models
        class MaxPoolCNN(nn.Module):
            def __init__(self):
                super(MaxPoolCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool2d(kernel_size=2)
                
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu2 = nn.ReLU()
                self.pool2 = nn.MaxPool2d(kernel_size=2)
                
                self.fc = nn.Linear(32*7*7, 10)
                
            def forward(self, x):
                x = self.relu1(self.bn1(self.conv1(x)))
                x = self.pool1(x)
                
                x = self.relu2(self.bn2(self.conv2(x)))
                x = self.pool2(x)
                
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        class AvgPoolCNN(nn.Module):
            def __init__(self):
                super(AvgPoolCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.AvgPool2d(kernel_size=2)
                
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu2 = nn.ReLU()
                self.pool2 = nn.AvgPool2d(kernel_size=2)
                
                self.fc = nn.Linear(32*7*7, 10)
                
            def forward(self, x):
                x = self.relu1(self.bn1(self.conv1(x)))
                x = self.pool1(x)
                
                x = self.relu2(self.bn2(self.conv2(x)))
                x = self.pool2(x)
                
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        class BasisPoolCNN(nn.Module):
            def __init__(self):
                super(BasisPoolCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.relu1 = nn.ReLU()
                self.pool1 = BasisPooling(kernel_size=2, basis_dim=2, temperature=0.1)
                
                # After basis pooling, channels are doubled
                self.conv2 = nn.Conv2d(16*2, 32, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu2 = nn.ReLU()
                self.pool2 = BasisPooling(kernel_size=2, basis_dim=1, temperature=0.1)
                # Using basis_dim=1 for the second layer to keep parameter count comparable
                
                self.fc = nn.Linear(32*7*7, 10)
                
            def forward(self, x):
                x = self.relu1(self.bn1(self.conv1(x)))
                x = self.pool1(x)
                
                x = self.relu2(self.bn2(self.conv2(x)))
                x = self.pool2(x)
                
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        # Initialize models
        max_model = MaxPoolCNN()
        avg_model = AvgPoolCNN()
        basis_model = BasisPoolCNN()
        
        # Define loss and optimizers
        criterion = nn.CrossEntropyLoss()
        max_optimizer = torch.optim.Adam(max_model.parameters(), lr=0.001)
        avg_optimizer = torch.optim.Adam(avg_model.parameters(), lr=0.001)
        basis_optimizer = torch.optim.Adam(basis_model.parameters(), lr=0.001)
        
        # Training function
        def train_model(model, optimizer, name="Model"):
            start_time = time.time()
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                print(f"{name} - Epoch {epoch+1}: Loss: {epoch_loss/len(loader):.4f}")
            
            accuracy = 100. * correct / total
            duration = time.time() - start_time
            print(f"{name} - Training complete - Accuracy: {accuracy:.2f}%, Time: {duration:.2f}s")
            
            return accuracy, duration
        
        # Train all models
        print("Training MaxPool CNN...")
        max_accuracy, max_time = train_model(max_model, max_optimizer, "MaxPool")
        
        print("\nTraining AvgPool CNN...")
        avg_accuracy, avg_time = train_model(avg_model, avg_optimizer, "AvgPool")
        
        print("\nTraining BasisPool CNN...")
        basis_accuracy, basis_time = train_model(basis_model, basis_optimizer, "BasisPool")
        
        # Compare results
        print("\n===== COMPARISON RESULTS =====")
        print(f"MaxPool:   Accuracy {max_accuracy:.2f}%, Training time {max_time:.2f}s")
        print(f"AvgPool:   Accuracy {avg_accuracy:.2f}%, Training time {avg_time:.2f}s")
        print(f"BasisPool: Accuracy {basis_accuracy:.2f}%, Training time {basis_time:.2f}s")
        
        # Count parameters in each model
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        max_params = count_parameters(max_model)
        avg_params = count_parameters(avg_model)
        basis_params = count_parameters(basis_model)
        
        print("\nParameter count:")
        print(f"MaxPool:   {max_params:,}")
        print(f"AvgPool:   {avg_params:,}")
        print(f"BasisPool: {basis_params:,}")
        
        return {
            'accuracy': {
                'max': max_accuracy,
                'avg': avg_accuracy,
                'basis': basis_accuracy
            },
            'time': {
                'max': max_time,
                'avg': avg_time,
                'basis': basis_time
            },
            'parameters': {
                'max': max_params,
                'avg': avg_params,
                'basis': basis_params
            }
        }
        
    except ImportError:
        print("Could not import torchvision. Please install it to run this test.")
        return None

if __name__ == "__main__":
    print("=== Testing Basic Pooling Operations ===")
    pooling_results = compare_pooling_methods()
    print("\n=== Testing with MNIST Dataset ===")
    mnist_results = compare_on_mnist(num_epochs=20)