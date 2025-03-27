import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasisPooling(nn.Module):
    """
    Pools each spatial patch into a single output via a learned combination
    of (max, min, mean). The final shape is (B, C, H_out, W_out), same as
    standard MaxPool2d or AvgPool2d.

    Key Points:
      1) Single pass over patches: we compute min, max, sum => mean.
      2) Gating via a small param block that yields weights for [max, min, mean].
      3) Optional grouping: if grouping < channels, we have separate gating
         for each group of channels. This can increase or decrease parameter count.

    Args:
        kernel_size (int or tuple): Pooling region size (kH, kW).
        stride (int or tuple, optional): Defaults to kernel_size.
        grouping (int): Number of channels per gating group.
            - If grouping = 1, each channel has its own gating vector
              -> more parameters, more flexible.
            - If grouping = C, all channels share a single gating vector
              -> minimal parameters, fully global gating.
            - If grouping = 8, then channels are split into groups of 8.
        init_alpha (float): Initial values for gating parameters. 0 => uniform gating.
    """
    def __init__(self, kernel_size, stride=None, grouping=8, init_alpha=0.0):
        super().__init__()

        # Handle kernel and stride
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride
        self.grouping = grouping  # how many channels share a gating vector?

        # We'll create gating params of shape (num_groups, 3).
        # Each row => alpha for [max, min, mean]. We'll softmax them per group.
        # The number of groups = 'group_count', computed after we know input shape.
        self.init_alpha = init_alpha

        # We don't know the exact # of channels until forward unless you pass it in at init.
        # So we'll lazily create them in forward if they aren't initialized.
        self.gate_params = None
        self.register_buffer("initialized", torch.tensor([False], dtype=torch.bool))

    def _init_params(self, C, device):
        """
        Create self.gate_params given the channel count.
        We'll do that once, on the first forward call.
        """
        # Number of groups
        group_count = (C + self.grouping - 1) // self.grouping  # ceiling division
        # Each group gets a 3-element alpha
        alpha_init = torch.full((group_count, 3), self.init_alpha, device=device)
        self.gate_params = nn.Parameter(alpha_init)
        self.initialized[0] = True

    def forward(self, x):
        """
        x: (B, C, H, W)

        returns: (B, C, H_out, W_out)
        """
        B, C, H, W = x.shape

        if not self.initialized[0]:
            self._init_params(C, x.device)

        # Calculate output spatial dimensions
        kH, kW = self.kernel_size
        sH, sW = self.stride
        out_height = (H - kH) // sH + 1
        out_width = (W - kW) // sW + 1
        N = out_height * out_width

        # ----------------------------------------------------
        # 1) Extract patches once: shape => (B, C, kH*kW, N)
        # ----------------------------------------------------
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        k_sq = kH * kW
        patches = patches.view(B, C, k_sq, N)

        # ----------------------------------------------------
        # 2) Single-pass stats: min, max, sum => mean
        # ----------------------------------------------------
        min_vals, _ = patches.min(dim=2)   # (B, C, N)
        max_vals, _ = patches.max(dim=2)   # (B, C, N)
        sum_vals = patches.sum(dim=2)      # (B, C, N)
        mean_vals = sum_vals / float(k_sq) # (B, C, N)

        # We'll gather them into a "basis stack" => shape (B, C, 3, N)
        basis_stack = torch.stack([max_vals, min_vals, mean_vals], dim=2)
        # basis_stack[b, c, i, n], i in {0,1,2} => {max, min, mean}

        # ----------------------------------------------------
        # 3) Apply gating. We'll do grouping on channels.
        #    For each group, we have a 3-element alpha => softmax => weights
        #    Then we multiply basis_stack by those weights.
        # ----------------------------------------------------
        # gate_params: (num_groups, 3)
        # We'll split channels into groups of size self.grouping each.

        # 3a) Softmax the gating parameters
        # shape => (num_groups, 3)
        # => weights => (num_groups, 3)
        gate_alphas = F.softmax(self.gate_params, dim=1)  # row-wise softmax

        # 3b) We'll tile or index the correct gating row for each channel
        # Let's build an index [group_idx for each channel]
        group_count = gate_alphas.shape[0]
        # e.g., if grouping=8 and we have C=32 => group_count=4 => channels 0..7 -> group0, 8..15->group1...
        channel_groups = torch.arange(C, device=x.device) // self.grouping
        # shape => (C,) with values in [0, group_count-1]

        # We'll gather the gating weights for each channel => (C, 3)
        # so we can broadcast them over B, N. 
        # First expand gate_alphas => (group_count, 1, 3, 1) then gather via index_select
        # or we do something simpler: gather row-wise.
        gating_per_channel = gate_alphas[channel_groups]  # => (C, 3)

        # We want to reshape so it matches the basis_stack shape (B, C, 3, N)
        # We'll do gating_per_channel => (1, C, 3, 1)
        gating_per_channel = gating_per_channel.unsqueeze(0).unsqueeze(-1)
        # => (1, C, 3, 1)

        # 3c) Weighted sum across the "3" dimension
        # basis_stack => (B, C, 3, N)
        # gating      => (1, C, 3, 1)
        # => multiply => (B, C, 3, N), sum over dim=2 => (B, C, N)
        weighted = basis_stack * gating_per_channel  # broadcast
        blended = weighted.sum(dim=2)  # => (B, C, N)

        # ----------------------------------------------------
        # 4) Reshape to final (B, C, out_height, out_width)
        # ----------------------------------------------------
        out = blended.view(B, C, out_height, out_width)
        return out





# Example usage in a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        
        # Use Basis Pooling instead of MaxPool
        self.pool1 = BasisPooling(kernel_size=2)
        
        # Note: After basis pooling, we have 16*2=32 channels
        self.conv2 = nn.Conv2d(16*2, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        
        # Another basis pooling layer
        self.pool2 = BasisPooling(kernel_size=2)
        
        # Fully connected layers
        self.fc = nn.Linear(32*2*8*8, num_classes)  # Assuming 32x32 input images
        
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
    basis_pool = BasisPooling(kernel_size=2, stride=2)
    
    # Forward passes
    max_output = max_pool(x)
    avg_output = avg_pool(x)
    basis_output = basis_pool(x)
    
    # # Print input and shapes
    # print("Input tensor:")
    # print(x[0, :, :, :])  # First batch item
    # print(f"Input shape: {x.shape}")
    
    # print("\nMax Pool output:")
    # print(max_output[0, :, :, :])
    # print(f"Max Pool output shape: {max_output.shape}")
    
    # print("\nAvg Pool output:")
    # print(avg_output[0, :, :, :])
    # print(f"Avg Pool output shape: {avg_output.shape}")
    
    # print("\nBasis Pool output (first half of channels):")
    # print(basis_output[0, :channels, :, :])
    # print("\nBasis Pool output (second half of channels):")
    # print(basis_output[0, channels:, :, :])
    # print(f"Basis Pool output shape: {basis_output.shape}")
    
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
                self.pool1 = BasisPooling(kernel_size=2)
                
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu2 = nn.ReLU()
                self.pool2 = BasisPooling(kernel_size=2)
                
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

# Uncomment to run tests
# pooling_results = compare_pooling_methods()
# mnist_results = compare_on_mnist(num_epochs=2)

if __name__ == "__main__":
    # print("=== Testing Basic Pooling Operations ===")
    # pooling_results = compare_pooling_methods()
    print("\n=== Testing with MNIST Dataset ===")
    mnist_results = compare_on_mnist(num_epochs=10)