import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt




def compute_covariance(features): 
    ######################
    n = features.size(0)
    mean = torch.mean(features, dim=0, keepdim=True)
    cov = (features - mean).T @ (features - mean) / (n - 1)
    return cov
    ######################
    


def train_baseline(model, source_loader, target_loader, args, device):
    """Standard source training"""
    print("\nTraining Baseline Model...")

    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    # Metrics storage
    metrics = {
        'source_loss': [],
        'target_loss': [],
        'source_accuracy': [],
        'target_accuracy': []
    }
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for data, target in tqdm(source_loader, desc=f'Epoch {epoch}'):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # TODO Calculate training and testing metrics
        source_loss, source_accuracy = evaluate(model, source_loader, device)
        target_loss, target_accuracy = evaluate(model, target_loader, device)

        # Store metrics for plotting
        metrics['source_loss'].append(source_loss)
        metrics['target_loss'].append(target_loss)
        metrics['source_accuracy'].append(source_accuracy)
        metrics['target_accuracy'].append(target_accuracy)

        # Print loss and accuracy for source and target 
        print(f"Epoch {epoch}: Total Loss: {total_loss:.4f}")
        print('Source Loss & Accuracy: ', source_loss, source_accuracy)
        print('Target Loss & Accuracy: ', target_loss, target_accuracy)

    # Save final model
    torch.save(model.state_dict(), 'final_baseline.pth')
    plot_metrics(metrics, 'Baseline Model')

    # Return Final target accuracy 
    return target_accuracy

def train_coral(model, source_loader, target_loader, args, device):
    """CORAL training"""
    print("\nTraining CORAL Model...")

    # Metrics storage
    metrics = {
        'source_loss': [],
        'target_loss': [],
        'source_accuracy': [],
        'target_accuracy': []
    }

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for (source_data, source_target), (target_data, _) in zip(source_loader, target_loader):

            source_data = source_data.to(device)
            source_target = source_target.to(device)
            target_data = target_data.to(device)
            
            optimizer.zero_grad()
            
            # Extract features
            source_features = model.feature_extractor(source_data)
            target_features = model.feature_extractor(target_data)
            
            # Classification loss
            source_outputs = model.classifier(source_features)
            cls_loss = F.nll_loss(source_outputs, source_target)
            
            # TODO CORAL loss
            source_cov = compute_covariance(source_features)
            target_cov = compute_covariance(target_features)
            coral_loss = torch.norm(source_cov - target_cov, p='fro') ** 2 / (4 * source_features.size(1) ** 2)
            
            # Total loss
            loss = cls_loss + args.coral_weight * coral_loss 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # TODO Calculate training and testing metrics
        source_loss, source_accuracy = evaluate(model, source_loader, device)
        target_loss, target_accuracy = evaluate(model, target_loader, device)

        # Store metrics for plotting
        metrics['source_loss'].append(source_loss)
        metrics['target_loss'].append(target_loss)
        metrics['source_accuracy'].append(source_accuracy)
        metrics['target_accuracy'].append(target_accuracy)
      
        # Print loss and accuracy for source and target 
        print(f"Epoch {epoch}: Source Loss {source_loss:.2f} Accuracy: {100 * source_accuracy:.2f}%")
        print(f"Target Loss {target_loss:.2f} Accuracy: {100 * target_accuracy:.2f}%")
    
    
    # Save final model
    torch.save(model.state_dict(), 'final_coral.pth')

    plot_metrics(metrics, 'Coral Model')

    # Return Final target accuracy 
    return target_accuracy

def train_adversarial(model, source_loader, target_loader, args, device):
    """Adversarial training"""
    print("\nTraining Adversarial Model...")
    
    # Metrics storage
    metrics = {
        'source_loss': [],
        'target_loss': [],
        'source_accuracy': [],
        'target_accuracy': []
    }

    discriminator = nn.Sequential(
        nn.Linear(256, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 2),
    ).to(device)
    
    optimizer_g = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        model.train()
        discriminator.train()
        total_loss_g = 0
        total_loss_d = 0
        
        for (source_data, source_target), (target_data, _) in zip(source_loader, target_loader):
            source_data = source_data.to(device)
            source_target = source_target.to(device)
            target_data = target_data.to(device)
            batch_size = source_data.size(0)
            
            # Train discriminator
            optimizer_d.zero_grad()
            
            source_features = model.feature_extractor(source_data).detach()
            target_features = model.feature_extractor(target_data).detach()
            
            source_domain = torch.zeros(batch_size).long().to(device)
            target_domain = torch.ones(batch_size).long().to(device)
            
            ###TODO###
            source_domain_pred = discriminator(source_features)
            target_domain_pred = discriminator(target_features)
            d_loss = F.cross_entropy(source_domain_pred, source_domain) + F.cross_entropy(target_domain_pred, target_domain)
            ####

            d_loss.backward()
            optimizer_d.step()
            total_loss_d += d_loss.item()
            
            # Train generator
            optimizer_g.zero_grad()
            
            source_features = model.feature_extractor(source_data)
            target_features = model.feature_extractor(target_data)
            source_outputs = model.classifier(source_features)
            
            # Classification loss
            cls_loss = F.nll_loss(source_outputs, source_target)
            
            ## TODO ## generator loss
            source_domain_pred = discriminator(source_features)
            target_domain_pred = discriminator(target_features)
            g_loss = F.cross_entropy(source_domain_pred, target_domain) + F.cross_entropy(target_domain_pred, source_domain)
            ####

            loss_g = cls_loss + args.adversarial_weight * g_loss
            loss_g.backward()
            optimizer_g.step()
            total_loss_g += loss_g.item()
        
        # TODO Calculate training and testing metrics
        source_loss, source_accuracy = evaluate(model, source_loader, device)
        target_loss, target_accuracy = evaluate(model, target_loader, device)

        # Store metrics for plotting
        metrics['source_loss'].append(source_loss)
        metrics['target_loss'].append(target_loss)
        metrics['source_accuracy'].append(source_accuracy)
        metrics['target_accuracy'].append(target_accuracy)
      
        # Print loss and accuracy for source and target 
        print(f"Epoch {epoch}: Source Loss {source_loss:.2f} Accuracy: {100 * source_accuracy:.2f}%")
        print(f"Target Loss {target_loss:.2f} Accuracy: {100 * target_accuracy:.2f}%")
    
    # Save final model
    torch.save({
        'model': model.state_dict(),
        'discriminator': discriminator.state_dict()
    }, 'final_adversarial.pth')

    plot_metrics(metrics, 'Adversarial Model')

    # Return Final target accuracy 
    return target_accuracy


def train_adabn(model, source_loader, target_loader, args, device):
    """AdaBN with source training and target adaptation"""
    print("\nTraining AdaBN Model...")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Metrics storage
    metrics = {
        'source_loss': [],
        'target_loss': [],
        'source_accuracy': [],
        'target_accuracy': []
    }

    # 1. Train on source
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for data, target in tqdm(source_loader, desc=f'Epoch {epoch} (Source Training)'):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

         # Store metrics for plotting
        metrics['source_loss'].append(source_loss)
        metrics['target_loss'].append(target_loss)
        metrics['source_accuracy'].append(source_accuracy)
        metrics['target_accuracy'].append(target_accuracy)
    
    # 2. Adapt BN statistics on target domain
    model.train()
    print("\nAdapting BN statistics on target domain...")
    

    ########################################################
    # Implement AdaBN (foward on target data for args.epoch)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.running_mean.zero_()
            module.running_var.fill_(1)
            module.num_batches_tracked.zero_()
            module.momentum = 1.0  # Set momentum to 1.0 for full adaptation

    # Run target data through model to update BN statistics
    with torch.no_grad():
        for data, _ in tqdm(target_loader, desc='Adapting BN'):
            data = data.to(device)
            model(data)
    ######################################################
    
    # TODO Calculate target accuracy and print it 
    source_loss, source_accuracy = evaluate(model, source_loader, device)
    target_loss, target_accuracy = evaluate(model, target_loader, device)
    
    # Print loss and accuracy for source and target 
    print(f"Epoch {epoch}: Source Loss {source_loss:.2f} Accuracy: {100 * source_accuracy:.2f}%")
    print(f"Target Loss {target_loss:.2f} Accuracy: {100 * target_accuracy:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 'final_adabn.pth')

    plot_metrics(metrics, 'AdaBN Model')

    # Return Final target accuracy 
    return target_accuracy

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target).item()
            pred = output.max(1)[1]
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(loader), correct / total


def plot_metrics(metrics, method_name):
    epochs = range(1, len(metrics['source_loss']) + 1)
    
    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['source_loss'], label='Source Loss')
    plt.plot(epochs, metrics['target_loss'], label='Target Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{method_name} Loss Across Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['source_accuracy'], label='Source Accuracy')
    plt.plot(epochs, metrics['target_accuracy'], label='Target Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{method_name} Accuracy Across Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return
