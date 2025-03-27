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


class DotPoolCNN(nn.Module):
    def __init__(self, basis_dim=2):
        super(DotPoolCNN, self).__init__()
        # Similar architecture but with DotPool and adjusted channels
        self.basis_dim = basis_dim
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = DotPooling(kernel_size=2, nonlinearity=None, stride=2)
        
        # After first basis pooling, channels are multiplied by basis_dim
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = DotPooling(kernel_size=2, nonlinearity=None, stride=2)
        
        # After second basis pooling, channels are again multiplied
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


def compare_on_cifar10(num_epochs=5):
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
        dot_model = DotPoolCNN()
        
        # Define loss and optimizers
        criterion = nn.CrossEntropyLoss()
        max_optimizer = torch.optim.Adam(max_model.parameters(), lr=0.001)
        avg_optimizer = torch.optim.Adam(avg_model.parameters(), lr=0.001)
        dot_optimizer = torch.optim.Adam(dot_model.parameters(), lr=0.001) 


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
        
        print("Training DotPool CNN...")
        dot_results = train_model(dot_model, max_optimizer, "DotPool")
        results['DotPool'] = dot_results
        
        # Count parameters
        param_counts = {
            'MaxPool': sum(p.numel() for p in max_model.parameters() if p.requires_grad),
            'AvgPool': sum(p.numel() for p in avg_model.parameters() if p.requires_grad),
            'DotPool': sum(p.numel() for p in dot_model.parameters() if p.requires_grad)
        }
               
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
    results, param_counts = compare_on_cifar10(num_epochs=5)