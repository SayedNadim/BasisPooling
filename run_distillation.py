import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from knowledge_distillation import (
    SimpleCNN, 
    StudentCNN, 
    load_mnist, 
    load_cifar10, 
    train_teacher, 
    train_student_with_distillation,
    evaluate_model,
    DistillationDataset
)
from torch.utils.data import DataLoader
import os

def plot_results(results):
    models = results['Model']
    mnist_acc = results['MNIST Accuracy']
    cifar_acc = results['CIFAR Accuracy']
    avg_acc = results['Average Accuracy']
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, mnist_acc, width, label='MNIST Accuracy', color='#3498db')
    rects2 = ax.bar(x, cifar_acc, width, label='CIFAR Accuracy', color='#e74c3c')
    rects3 = ax.bar(x + width, avg_acc, width, label='Average Accuracy', color='#2ecc71')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Set y-axis range and add grid lines
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Function to add labels to bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', 
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Vertical offset in points
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    
    plt.tight_layout()
    plt.show()

def run_distillation_experiment(device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load datasets
    mnist_train_loader, mnist_test_loader, mnist_train_dataset, mnist_test_dataset = load_mnist()
    cifar_train_loader, cifar_test_loader, cifar_train_dataset, cifar_test_dataset = load_cifar10()
    
    # Initialize teacher models
    teacher_mnist = SimpleCNN(num_classes=10, input_channels=3)  # Using 3 channels to match CIFAR
    teacher_cifar = SimpleCNN(num_classes=10, input_channels=3)
    
    # Train teacher models (or load pre-trained ones if available)
    try:
        teacher_mnist.load_state_dict(torch.load('teacher_mnist.pth'))
        print("Loaded pre-trained MNIST teacher model")
    except:
        print("Training MNIST teacher model...")
        teacher_mnist = train_teacher(teacher_mnist, mnist_train_loader, epochs=5, device=device, task_name="MNIST")
        torch.save(teacher_mnist.state_dict(), 'teacher_mnist.pth')
    
    try:
        teacher_cifar.load_state_dict(torch.load('teacher_cifar.pth'))
        print("Loaded pre-trained CIFAR teacher model")
    except:
        print("Training CIFAR teacher model...")
        teacher_cifar = train_teacher(teacher_cifar, cifar_train_loader, epochs=5, device=device, task_name="CIFAR")
        torch.save(teacher_cifar.state_dict(), 'teacher_cifar.pth')
    
    # Evaluate teacher models
    print("\n===== Evaluating Teacher Models =====")
    print("MNIST teacher on MNIST:")
    mnist_teacher_acc = evaluate_model(teacher_mnist, mnist_test_loader, device=device)
    print(f"Accuracy: {mnist_teacher_acc:.2f}%")
    
    print("CIFAR teacher on CIFAR:")
    cifar_teacher_acc = evaluate_model(teacher_cifar, cifar_test_loader, device=device)
    print(f"Accuracy: {cifar_teacher_acc:.2f}%")
    
    # Create distillation dataset
    distill_dataset = DistillationDataset(mnist_train_dataset, cifar_train_dataset)
    distill_loader = DataLoader(distill_dataset, batch_size=64, shuffle=True)
    
    # Initialize student model (with larger capacity)
    student_model = StudentCNN(num_classes=10, input_channels=3)
    
    # Hyperparameters for distillation
    temp = 4.0   # Temperature for softening distributions
    alpha = 0.5  # Weight for student loss (hard targets)
    beta = 0.5   # Weight for distillation loss (soft targets)
    epochs = 15  # More epochs for better learning
    
    # Train student with knowledge distillation
    print("\n===== Training Student Model with Knowledge Distillation =====")
    student_model = train_student_with_distillation(
        student_model, teacher_mnist, teacher_cifar,
        distill_loader, mnist_test_loader, cifar_test_loader,
        epochs=epochs, temp=temp, alpha=alpha, beta=beta,
        device=device
    )
    
    # Save the trained student model
    torch.save(student_model.state_dict(), 'results/student_model.pth')
    
    # Final evaluation
    print("\n===== Final Evaluation =====")
    print("Student model on MNIST:")
    mnist_student_acc = evaluate_model(student_model, mnist_test_loader, device=device)
    print(f"Accuracy: {mnist_student_acc:.2f}%")
    
    print("Student model on CIFAR:")
    cifar_student_acc = evaluate_model(student_model, cifar_test_loader, device=device)
    print(f"Accuracy: {cifar_student_acc:.2f}%")
    
    # Compare with teacher models
    results = {
        'Model': ['MNIST Teacher', 'CIFAR Teacher', 'Student'],
        'MNIST Accuracy': [mnist_teacher_acc, 0, mnist_student_acc],
        'CIFAR Accuracy': [0, cifar_teacher_acc, cifar_student_acc],
        'Average Accuracy': [mnist_teacher_acc/2, cifar_teacher_acc/2, (mnist_student_acc + cifar_student_acc)/2]
    }
    
    # Plot results
    plot_results(results)
    
    print("\n===== Experiment Summary =====")
    print(f"MNIST Teacher Accuracy: {mnist_teacher_acc:.2f}%")
    print(f"CIFAR Teacher Accuracy: {cifar_teacher_acc:.2f}%")
    print(f"Student Model MNIST Accuracy: {mnist_student_acc:.2f}%")
    print(f"Student Model CIFAR Accuracy: {cifar_student_acc:.2f}%")
    print(f"Student Model Average Accuracy: {(mnist_student_acc + cifar_student_acc)/2:.2f}%")
    
    # Test how well the student generalizes across domains
    print("\n===== Cross-Domain Evaluation =====")
    mnist_on_cifar = evaluate_model(teacher_mnist, cifar_test_loader, device=device)
    cifar_on_mnist = evaluate_model(teacher_cifar, mnist_test_loader, device=device)
    
    print(f"MNIST Teacher on CIFAR: {mnist_on_cifar:.2f}% (Cross-domain)")
    print(f"CIFAR Teacher on MNIST: {cifar_on_mnist:.2f}% (Cross-domain)")
    
    return student_model, results

if __name__ == "__main__":
    run_distillation_experiment()
