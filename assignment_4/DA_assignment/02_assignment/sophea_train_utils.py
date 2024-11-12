import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np
from tqdm import tqdm



def compute_covariance(features): 
    ######################
    # Implement Coral loss
    ######################
    return # ... Fill


def train_baseline(model, source_loader, target_loader, args, device):
    """Standard source training"""
    print("\nTraining Baseline Model...")

    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
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
        
        # Calculate training and testing metrics
        # ..... Fill 
        # Print loss and accuracy for source and target 
        
    # Save final model
    torch.save(model.state_dict(), 'final_baseline.pth')

    # Return Final target accuracy 
    return # ..... Fill 

def train_coral(model, source_loader, target_loader, args, device):
    """CORAL training"""
    print("\nTraining CORAL Model...")

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
            
            # CORAL loss
            coral_loss = ... # ..... Fill
            
            # Total loss
            loss = cls_loss + args.coral_weight * coral_loss 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate training and testing metrics
        # ..... Fill 
        # Print loss and accuracy for source and target 
    
    # Save final model
    torch.save(model.state_dict(), 'final_coral.pth')

    # Return Final target accuracy 
    return # ..... Fill 

def train_adversarial(model, source_loader, target_loader, args, device):
    """Adversarial training"""
    print("\nTraining Adversarial Model...")
    
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
            
            source_domain_pred = ... # ... Fill
            target_domain_pred = ... # ... Fill 
            
            d_loss = ... #F.cross_entropy(...,...) + F.cross_entropy(...,...) Fill
            
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
            
            # Adversarial loss

            source_domain_pred = ... # ... Fill 
            target_domain_pred = ... # ... Fill  
            
            g_loss = ... #F.cross_entropy(...,...) + F.cross_entropy(...,...) Fill
            
            loss_g = cls_loss + args.adversarial_weight * g_loss
            loss_g.backward()
            optimizer_g.step()
            total_loss_g += loss_g.item()
        
        # Calculate training and testing metrics
        # ..... Fill 
        # Print loss and accuracy for source and target 
    
    # Save final model
    torch.save({
        'model': model.state_dict(),
        'discriminator': discriminator.state_dict()
    }, 'final_adversarial.pth')

    # Return Final target accuracy 
    return # ..... Fill 


def train_adabn(model, source_loader, target_loader, args, device):
    """AdaBN with source training and target adaptation"""
    print("\nTraining AdaBN Model...")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
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
    
    # 2. Adapt BN statistics on target domain
    model.train()
    print("\nAdapting BN statistics on target domain...")
    

    ########################################################
    # Implement AdaBN (foward on target data for args.epoch)
    ######################################################
    
    # Calculate target accuracy and print it 
    # ..... Fill 
    
    # Save final model
    torch.save(model.state_dict(), 'final_adabn.pth')

    # Return Final target accuracy 
    return # ..... Fill 

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