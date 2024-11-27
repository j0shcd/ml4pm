import argparse
import torch
import numpy as np
import random
from model import BaselineModel
from dataloader import prepare_data
from train_utils import train_baseline, train_coral, train_adversarial, train_adabn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.source_loader, self.target_loader = prepare_data(batch_size=args.batch_size)

    def train_baseline(self):
        model = BaselineModel().to(self.device)
        return train_baseline(model, self.source_loader, self.target_loader, 
                            self.args, self.device)

    def train_coral(self):
        model = BaselineModel().to(self.device)
        return train_coral(model, self.source_loader, self.target_loader, 
                          self.args, self.device)

    def train_adversarial(self):
        model = BaselineModel().to(self.device)
        return train_adversarial(model, self.source_loader, self.target_loader, 
                               self.args, self.device)

    def train_adabn(self):
        model = BaselineModel().to(self.device)
        return train_adabn(model, self.source_loader, self.target_loader, 
                          self.args, self.device)
        
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(args.seed)
    trainer = Trainer(args, device)
    
    results = {}
    
    if args.method == 'baseline' or args.method == 'all':
        results['baseline'] = trainer.train_baseline()
    
    if args.method == 'coral' or args.method == 'all':
        results['coral'] = trainer.train_coral()
    
    if args.method == 'adversarial' or args.method == 'all':
        results['adversarial'] = trainer.train_adversarial()
    
    if args.method == 'adabn' or args.method == 'all':
        results['adabn'] = trainer.train_adabn()
    
    print("\nFinal Target Accuracies:")
    for method, acc in results.items():
        print(f"{method}: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation Methods')
    parser.add_argument('--method', type=str, default='all',
                        choices=['baseline', 'coral', 'adversarial', 'adabn', 'all'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--coral_weight', type=float, default=1.0)
    parser.add_argument('--adversarial_weight', type=float, default=1.0)
    args = parser.parse_args()
    
    main(args)