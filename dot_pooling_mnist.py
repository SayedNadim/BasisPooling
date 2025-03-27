import torch
import torch.nn as nn
import torch.nn.functional as F

class DotPooling(nn.Module):   
    def __init__(self, kernel_size, stride=None, nonlinearity = None, eps=1e-6):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        
        # Compute output spatial dimensions.
        out_h = (H - kH) // sH + 1
        out_w = (W - kW) // sW + 1
        
        # Unfold the input into patches.
        # Shape: (B, C*kH*kW, N) with N = out_h*out_w.
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        # Reshape to (B, C, kH*kW, N) so that each channel's patch is separate.
        patches = patches.view(B, C, kH * kW, -1)
        
        # Compute patch average: (B, C, N)
        p_avg = patches.mean(dim=2)
        # Compute patch L2 norm: (B, C, N)
        p_norm = patches.pow(2).sum(dim=2).sqrt() + self.eps
        
        # The norm of an all-ones vector of length n = kH*kW.
        ref_norm = (kH * kW) ** 0.5
        # Cosine similarity per patch:
        # Note: patches.sum(dim=2) == (kH*kW) * p_avg.
        cos_sim = (kH * kW * p_avg) / (p_norm * ref_norm + self.eps)
        # Simplifies to: cos_sim = (sqrt(n) * p_avg) / (p_norm + eps)
        
        # Adaptive output: weight the patch average by its cosine similarity.
        out_val = cos_sim * p_avg
        
        # Reshape to (B, C, out_h, out_w)
        out_val = out_val.view(B, C, out_h, out_w)
        return out_val



class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        
        self.pool1 = DotPooling(kernel_size=2, nonlinearity=None, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        
        self.pool2 = DotPooling(kernel_size=2, nonlinearity=None, stride=2)
        
        # Fully connected => now input channels=128, 
        # if input was e.g. 32x32, after two 2x2 pools => 8x8 => 32 * 8 * 8
        self.fc = nn.Linear(32*8*8, num_classes)
        
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
    basis_pool = DotPooling(kernel_size=2, nonlinearity=None, stride=2)
    
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
    
    print("\Dot Pool output:")
    print(basis_output[0, :, :, :])
    
    # Check for information preservation
    # Try reconstructing the original pattern from the pooled outputs
    
    # Simplistic reconstruction (upsampling by repeating values)
    def simple_upsample(tensor):
        return tensor.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
    
    max_reconstruction = simple_upsample(max_output)
    avg_reconstruction = simple_upsample(avg_output)
    basis_reconstruction = simple_upsample(basis_output)
    
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
                self.pool1 = DotPooling(kernel_size=2, nonlinearity=None, stride=2)
                
                # After basis pooling, channels are doubled
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu2 = nn.ReLU()
                self.pool2 = DotPooling(kernel_size=2, nonlinearity=None, stride=2)
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