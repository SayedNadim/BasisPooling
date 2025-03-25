import torch
import torch.nn as nn
import torch.nn.functional as F


class FastBasisPooling(nn.Module):
    """
    Ultra-fast implementation of Basis Pooling that approximates the selection
    mechanism with minimal computational overhead.
    """
    def __init__(self, kernel_size, stride=None, basis_dim=2):
        super(FastBasisPooling, self).__init__()
        
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
            # Second basis: min pooling for complementary information
            min_values, _ = patches.min(dim=2)
            outputs.append(min_values)
        
        # Additional bases: statistical moments
        if self.basis_dim > 2:
            # Third basis: mean values
            mean_values = patches.mean(dim=2)
            outputs.append(mean_values)
            
        if self.basis_dim > 3:
            # Fourth: variance
            variance = ((patches - patches.mean(dim=2, keepdim=True))**2).mean(dim=2)
            outputs.append(variance)
            
        if self.basis_dim > 4:
            # Fifth and beyond: additional statistical features or spatial patterns
            for i in range(4, self.basis_dim):
                if i == 4:
                    # Median
                    moment, _ = patches.median(dim=2)
                else:
                    # Higher moments (approximated)
                    power = (i - 3) / 2 + 1.5
                    moment = ((patches - patches.mean(dim=2, keepdim=True))**power).mean(dim=2)
                
                outputs.append(moment)
        
        # Combine all basis vectors
        output = torch.stack(outputs, dim=2)
        output = output.view(batch_size, channels * self.basis_dim, out_height, out_width)
        
        return output


# CIFAR-10 CNN models
class MaxPoolCNN(nn.Module):
    def __init__(self):
        super(MaxPoolCNN, self).__init__()
        # Larger network for CIFAR-10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class AvgPoolCNN(nn.Module):
    def __init__(self):
        super(AvgPoolCNN, self).__init__()
        # Same architecture as MaxPoolCNN but with AvgPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class BasisPoolCNN(nn.Module):
    def __init__(self, basis_dim=2):
        super(BasisPoolCNN, self).__init__()
        # Similar architecture but with BasisPool and adjusted channels
        self.basis_dim = basis_dim
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = FastBasisPooling(kernel_size=2, basis_dim=basis_dim)
        
        # After first basis pooling, channels are multiplied by basis_dim
        self.conv3 = nn.Conv2d(32*basis_dim, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = FastBasisPooling(kernel_size=2, basis_dim=basis_dim)
        
        # After second basis pooling, channels are again multiplied
        self.fc1 = nn.Linear(64 * basis_dim * 8 * 8, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


def compare_on_cifar10(basis_dims=[2], num_epochs=5):
    try:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        import time
        
        # Load CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        
        # Use a subset for faster testing
        train_subset_indices = torch.randperm(len(train_dataset))[:5000]
        train_subset = torch.utils.data.Subset(train_dataset, train_subset_indices)
        
        test_subset_indices = torch.randperm(len(test_dataset))[:1000]
        test_subset = torch.utils.data.Subset(test_dataset, test_subset_indices)
        
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=100, shuffle=False)
        
        # Initialize baseline models
        max_model = MaxPoolCNN()
        avg_model = AvgPoolCNN()
        
        # Initialize basis models for each dimension to test
        basis_models = {}
        for basis_dim in basis_dims:
            basis_models[basis_dim] = BasisPoolCNN(basis_dim=basis_dim)
        
        # Define loss and optimizers
        criterion = nn.CrossEntropyLoss()
        max_optimizer = torch.optim.Adam(max_model.parameters(), lr=0.001)
        avg_optimizer = torch.optim.Adam(avg_model.parameters(), lr=0.001)
        
        basis_optimizers = {}
        for basis_dim in basis_dims:
            basis_optimizers[basis_dim] = torch.optim.Adam(
                basis_models[basis_dim].parameters(), lr=0.001)
        
        # Training function
        def train_model(model, optimizer, name="Model"):
            start_time = time.time()
            model.train()
            
            for epoch in range(num_epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    # Calculate training accuracy
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                    if batch_idx % 20 == 0:
                        print(f'{name} - Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                              f'Loss: {running_loss/(batch_idx+1):.4f}, '
                              f'Acc: {100.*correct/total:.2f}%')
                
            duration = time.time() - start_time
            print(f"{name} - Training complete - Time: {duration:.2f}s")
            
            # Test accuracy
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            test_accuracy = 100. * correct / total
            print(f'{name} - Test Accuracy: {test_accuracy:.2f}%')
            
            return {
                'training_time': duration,
                'test_accuracy': test_accuracy
            }
        
        # Train all models
        results = {}
        
        print("Training MaxPool CNN...")
        max_results = train_model(max_model, max_optimizer, "MaxPool")
        results['MaxPool'] = max_results
        
        print("\nTraining AvgPool CNN...")
        avg_results = train_model(avg_model, avg_optimizer, "AvgPool")
        results['AvgPool'] = avg_results
        
        # Train all basis models
        for basis_dim in basis_dims:
            print(f"\nTraining BasisPool CNN with K={basis_dim}...")
            basis_results = train_model(
                basis_models[basis_dim], 
                basis_optimizers[basis_dim], 
                f"BasisPool-K{basis_dim}"
            )
            results[f'BasisPool-K{basis_dim}'] = basis_results
        
        # Count parameters
        param_counts = {
            'MaxPool': sum(p.numel() for p in max_model.parameters() if p.requires_grad),
            'AvgPool': sum(p.numel() for p in avg_model.parameters() if p.requires_grad)
        }
        
        for basis_dim in basis_dims:
            param_counts[f'BasisPool-K{basis_dim}'] = sum(
                p.numel() for p in basis_models[basis_dim].parameters() if p.requires_grad)
        
        # Print comparison results
        print("\n===== COMPARISON RESULTS =====")
        for name, result in results.items():
            print(f"{name:12s}: Accuracy {result['test_accuracy']:.2f}%, "
                  f"Training time {result['training_time']:.2f}s, "
                  f"Parameters {param_counts[name]:,}")
        
        return results, param_counts
        
    except ImportError:
        print("Could not import torchvision. Please install it to run this test.")
        return None, None


if __name__ == "__main__":
    # Test with various K values
    basis_dims = [2, 3, 4]  # Test K=2, K=3, and K=4
    results, param_counts = compare_on_cifar10(basis_dims=basis_dims, num_epochs=5)