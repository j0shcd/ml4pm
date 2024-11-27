import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def fft_transform(sig):
    return np.abs(np.fft.fft(sig)[:len(sig)//2])

class VibrationDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def prepare_data(batch_size=128):
    # Load and preprocess data
    data_src = np.load("data/X.npy")[::10]
    label_src = np.load("data/y.npy")[::10]
    data_tgt = np.load("data/X-test.npy")[::10]
    label_tgt = np.load("data/y-test.npy")[::10]

    # Apply FFT
    data_src_fft = np.array([fft_transform(sig) for sig in data_src])
    data_tgt_fft = np.array([fft_transform(sig) for sig in data_tgt])

    # Expand dimensions
    data_src_fft = np.expand_dims(data_src_fft, axis=1)  # Channel dimension
    data_tgt_fft = np.expand_dims(data_tgt_fft, axis=1)

    # Create datasets
    source_dataset = VibrationDataset(data_src_fft, label_src)
    target_dataset = VibrationDataset(data_tgt_fft, label_tgt)

    # Create dataloaders
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    
    return source_loader, target_loader
