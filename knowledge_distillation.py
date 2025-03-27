import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import copy
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a CNN model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(SimpleCNN, self).__init__()
        self.input_channels = input_channels
        
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

    def forward(self, x):
        # Handle inputs with different channel dimensions
        if x.size(1) != self.input_channels and self.input_channels == 3 and x.size(1) == 1:
            # Convert grayscale to RGB by duplicating the channel
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) != self.input_channels and self.input_channels == 1 and x.size(1) == 3:
            # Convert RGB to grayscale by averaging channels
            x = x.mean(dim=1, keepdim=True)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feature_vector = x  # Store feature vector before classification
        x = self.classifier(x)
        return x, feature_vector

    def get_features(self, x):
        """Extract features without classification"""
        # Handle inputs with different channel dimensions
        if x.size(1) != self.input_channels and self.input_channels == 3 and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) != self.input_channels and self.input_channels == 1 and x.size(1) == 3:
            x = x.mean(dim=1, keepdim=True)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

# Define StudentCNN with capacity for both domains
class StudentCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(StudentCNN, self).__init__()
        self.input_channels = input_channels
        
        # Increased capacity in the student model
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 48, kernel_size=3, stride=1, padding=1),  # More filters
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),  # More filters
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),  # More filters
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Linear(192 * 4 * 4, 768),  # Wider FC layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(768, num_classes)
        )
        
        # Domain classification branch (optional)
        self.domain_classifier = nn.Sequential(
            nn.Linear(192 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # 2 domains: MNIST and CIFAR
        )

    def forward(self, x):
        # Handle inputs with different channel dimensions
        if x.size(1) != self.input_channels and self.input_channels == 3 and x.size(1) == 1:
            # Convert grayscale to RGB by duplicating the channel
            x = x.repeat(1, 3, 1, 1)
            
        features = self.features(x)
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        
        # Main task classification
        logits = self.classifier(flattened)
        
        # Domain classification (optional)
        domain_logits = self.domain_classifier(flattened)
        
        return logits, domain_logits, flattened

# Load datasets
def load_mnist(batch_size=64):
    transform = transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),  # Resize to match CIFAR dimensions
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset

def load_cifar10(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset

# Create a combined dataset for distillation
class DistillationDataset(Dataset):
    def __init__(self, mnist_dataset, cifar_dataset, mnist_size=None, cifar_size=None):
        self.mnist_dataset = mnist_dataset
        self.cifar_dataset = cifar_dataset
        
        # Use subsets if sizes are specified
        if mnist_size is not None:
            mnist_indices = torch.randperm(len(mnist_dataset))[:mnist_size]
            self.mnist_indices = mnist_indices
        else:
            self.mnist_indices = torch.arange(len(mnist_dataset))
            
        if cifar_size is not None:
            cifar_indices = torch.randperm(len(cifar_dataset))[:cifar_size]
            self.cifar_indices = cifar_indices
        else:
            self.cifar_indices = torch.arange(len(cifar_dataset))
            
        self.mnist_size = len(self.mnist_indices)
        self.cifar_size = len(self.cifar_indices)
        
    def __len__(self):
        return self.mnist_size + self.cifar_size
    
    def __getitem__(self, idx):
        # Choose MNIST or CIFAR based on index
        if idx < self.mnist_size:
            data, label = self.mnist_dataset[self.mnist_indices[idx]]
            domain = 0  # MNIST domain label
        else:
            cifar_idx = idx - self.mnist_size
            data, label = self.cifar_dataset[self.cifar_indices[cifar_idx]]
            domain = 1  # CIFAR domain label
            
        return data, label, domain

# Train a teacher model on a dataset
def train_teacher(model, train_loader, epochs=5, device='cuda', task_name=""):
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
            outputs, _ = model(inputs)  # Ignore feature vector return
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 99:
                print(f'{task_name} Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}, '
                      f'Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.3f}%')
                running_loss = 0.0
    
    return model

# Evaluation function
def evaluate_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if isinstance(model, StudentCNN):
                outputs, _, _ = model(inputs)
            else:
                outputs, _ = model(inputs)
                
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def train_student_with_distillation(student_model, teacher_mnist, teacher_cifar, 
                                    distill_loader, mnist_test_loader, cifar_test_loader,
                                    epochs=10, temp=2.0, alpha=0.7, beta=0.3, gamma=0.5,
                                    device='cuda'):
    student_model.to(device)
    teacher_mnist.to(device)
    teacher_cifar.to(device)
    
    # Set teacher models to evaluation mode
    teacher_mnist.eval()
    teacher_cifar.eval()
    
    # Determine feature dimensions:
    # Teacher (SimpleCNN) outputs 128*4*4 = 2048 features.
    teacher_feature_dim = 128 * 4 * 4  # 2048
    # Student (StudentCNN) outputs 192*4*4 = 3072 features.
    student_feature_dim = 192 * 4 * 4  # 3072
    
    # Create a projection layer to map teacher features to student feature space.
    projection = nn.Linear(teacher_feature_dim, student_feature_dim).to(device)
    
    # Loss functions
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    domain_loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer includes both the student model and the projection parameters.
    optimizer = optim.Adam(list(student_model.parameters()) + list(projection.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    best_avg_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets, domains) in enumerate(distill_loader):
            inputs, targets, domains = inputs.to(device), targets.to(device), domains.to(device)
            optimizer.zero_grad()
            
            # Forward pass through student model
            student_logits, domain_logits, student_features = student_model(inputs)
            
            # Compute teacher outputs and features based on domain
            with torch.no_grad():
                # Masks for MNIST (domain 0) and CIFAR (domain 1)
                mnist_indices = (domains == 0)
                cifar_indices = (domains == 1)
                
                teacher_logits = torch.zeros_like(student_logits)
                teacher_features = torch.zeros_like(student_features)
                
                if mnist_indices.sum() > 0:
                    mnist_inputs = inputs[mnist_indices]
                    mnist_teacher_logits, mnist_features = teacher_mnist(mnist_inputs)
                    teacher_logits[mnist_indices] = mnist_teacher_logits
                    teacher_features[mnist_indices] = projection(mnist_features)
                
                if cifar_indices.sum() > 0:
                    cifar_inputs = inputs[cifar_indices]
                    cifar_teacher_logits, cifar_features = teacher_cifar(cifar_inputs)
                    teacher_logits[cifar_indices] = cifar_teacher_logits
                    teacher_features[cifar_indices] = projection(cifar_features)
                
                # Soft targets with temperature scaling
                soft_targets = torch.nn.functional.softmax(teacher_logits / temp, dim=1)
            
            # Student's log-softmax outputs (for KL divergence)
            soft_student = torch.nn.functional.log_softmax(student_logits / temp, dim=1)
            distillation_loss = kl_div_loss(soft_student, soft_targets) * (temp * temp)
            
            # Cross-entropy loss with hard targets
            student_loss = ce_loss(student_logits, targets)
            
            # Feature matching loss (mean squared error)
            feature_loss = mse_loss(student_features, teacher_features)
            
            # Domain classification loss (for the optional branch)
            domain_classification_loss = domain_loss_fn(domain_logits, domains)
            
            # Combined loss
            loss = (alpha * student_loss + 
                    beta * distillation_loss + 
                    gamma * feature_loss + 
                    (1 - gamma) * domain_classification_loss)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 49:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}, '
                      f'Loss: {running_loss/50:.3f}, Acc: {100.*correct/total:.3f}%')
                running_loss = 0.0
        
        # Evaluate the student model on test sets
        student_model.eval()
        mnist_acc = evaluate_model(student_model, mnist_test_loader, device)
        cifar_acc = evaluate_model(student_model, cifar_test_loader, device)
        avg_acc = (mnist_acc + cifar_acc) / 2
        
        print(f'Epoch {epoch+1}: MNIST Acc: {mnist_acc:.2f}%, CIFAR Acc: {cifar_acc:.2f}%, Avg: {avg_acc:.2f}%')
        
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_model_state = copy.deepcopy(student_model.state_dict())
            print(f'New best model with average accuracy: {best_avg_acc:.2f}%')
        
        scheduler.step(loss)
    
    # Load best model state if available
    if best_model_state is not None:
        student_model.load_state_dict(best_model_state)
        print(f'Loaded best model with average accuracy: {best_avg_acc:.2f}%')
    
    return student_model
