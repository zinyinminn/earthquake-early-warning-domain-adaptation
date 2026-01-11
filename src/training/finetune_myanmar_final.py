# finetune_myanmar_final.py


import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py


from myanmar_dataloader_h5 import MyanmarH5Dataset, wf_to_spec

print("="*70)
print("MYANMAR EARTHQUAKE DETECTION - FINAL TRAINING (>90% TARGET)")
print("="*70)

# -----------------------------------------------------------------
# 1. SMART DATA AUGMENTATION
# -----------------------------------------------------------------
class MyanmarAugmentor:
    """Advanced augmentation for seismic data"""
    
    @staticmethod
    def add_stationary_noise(waveform, noise_level=0.1):
        """Add realistic stationary noise"""
        noise = np.random.normal(0, noise_level, waveform.shape)
        return waveform + noise
    
    @staticmethod
    def time_warp(waveform, factor_range=(0.9, 1.1)):
        """Subtle time warping"""
        from scipy.interpolate import interp1d
        
        factor = np.random.uniform(factor_range[0], factor_range[1])
        old_length = waveform.shape[1]
        new_length = int(old_length * factor)
        
        warped = []
        for ch in range(waveform.shape[0]):
            x_old = np.linspace(0, 1, old_length)
            x_new = np.linspace(0, 1, new_length)
            f = interp1d(x_old, waveform[ch], kind='cubic')
            warped_ch = f(x_new)
            
            # Pad or truncate to original length
            if len(warped_ch) > old_length:
                warped_ch = warped_ch[:old_length]
            else:
                warped_ch = np.pad(warped_ch, (0, old_length - len(warped_ch)), 'edge')
            
            warped.append(warped_ch)
        
        return np.stack(warped)
    
    @staticmethod
    def channel_swap(waveform):
        """Swap channels (simulate different orientations)"""
        # Randomly permute the 3 channels
        perm = np.random.permutation(3)
        return waveform[perm]
    
    @staticmethod
    def frequency_mask(spectrogram, max_mask_percent=0.2):
        """Mask random frequency bands (simulate band noise)"""
        c, f, t = spectrogram.shape
        mask_height = int(f * np.random.uniform(0.05, max_mask_percent))
        mask_start = np.random.randint(0, f - mask_height)
        spectrogram[:, mask_start:mask_start+mask_height, :] = 0
        return spectrogram
    
    @staticmethod
    def apply_augmentations(waveform, is_training=True, prob=0.7):
        """Apply random augmentations if training"""
        if not is_training:
            return waveform
        
        augmented = waveform.copy()
        
        # Apply each augmentation with probability
        if np.random.random() < prob:
            augmented = MyanmarAugmentor.add_stationary_noise(augmented)
        
        if np.random.random() < prob/2:
            augmented = MyanmarAugmentor.time_warp(augmented)
        
        if np.random.random() < prob/3:
            augmented = MyanmarAugmentor.channel_swap(augmented)
        
        return augmented

# -----------------------------------------------------------------
# 2. ENHANCED DATASET WITH AUGMENTATION
# -----------------------------------------------------------------
class EnhancedMyanmarDataset(Dataset):
    """Myanmar dataset with smart augmentation"""
    
    def __init__(self, csv_path, h5_path, indices=None, 
                 augment=True, bandpass=(1.0, 20.0)):
        
        self.df = pd.read_csv(csv_path)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        
        self.h5_path = h5_path
        self.augment = augment
        self.bandpass = bandpass
        
        # Get valid trace names
        with h5py.File(h5_path, 'r') as f:
            h5_keys = set(f.keys())
        
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            trace_name = str(row['trace_name'])
            if trace_name in h5_keys:
                self.valid_indices.append(idx)
        
        print(f"Loaded {len(self.valid_indices)} valid samples")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        data_idx = self.valid_indices[idx]
        row = self.df.iloc[data_idx]
        
        trace_name = str(row['trace_name'])
        label = int(row['label_eq'])
        
        # Load waveform
        with h5py.File(self.h5_path, 'r') as f:
            waveform = f[trace_name][:]
        
        # Apply augmentation if training
        waveform = MyanmarAugmentor.apply_augmentations(
            waveform, is_training=self.augment
        )
        
        # Convert to spectrogram
        spec = wf_to_spec(waveform, fs=100.0)
        
        # Additional spectrogram augmentation
        if self.augment and np.random.random() < 0.3:
            spec = MyanmarAugmentor.frequency_mask(spec)
        
        # Convert to tensor
        X = torch.FloatTensor(spec)
        y = torch.LongTensor([label])
        
        return X, y

# -----------------------------------------------------------------
# 3. ENHANCED MODEL ARCHITECTURE
# -----------------------------------------------------------------
class MyanmarCNNEnhanced(nn.Module):
    """Enhanced CNN with attention mechanism"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Feature extractor with more capacity
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)  # 32x64x64
        x = self.conv2(x)  # 64x32x32
        x = self.conv3(x)  # 128x16x16
        x = self.conv4(x)  # 256x1x1
        
        # Squeeze spatial dimensions
        x = x.view(x.size(0), -1)  # (batch, 256)
        
        # Apply attention
        attn_weights = self.attention(x)
        x = x * attn_weights
        
        # Classify
        x = self.classifier(x)
        return x

# -----------------------------------------------------------------
# 4. TRAINING WITH FOCAL LOSS
# -----------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss to handle class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# -----------------------------------------------------------------
# 5. MAIN TRAINING FUNCTION
# -----------------------------------------------------------------
def main():
    # Paths
    CSV_PATH = r"D:\datasets\myanmar_eq\myanmar_full.csv"
    H5_PATH = r"D:\datasets\myanmar_eq\myanmar_full.hdf5"
    OUTPUT_DIR = r"C:\Users\USER\eew\models\myanmar_final"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = pd.read_csv(CSV_PATH)
    y = df['label_eq'].values
    indices = np.arange(len(df))
    
    # Stratified split
    i_train, i_temp = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
    y_temp = y[i_temp]
    i_val, i_test = train_test_split(i_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Dataset split:")
    print(f"  Train: {len(i_train)} samples")
    print(f"  Val:   {len(i_val)} samples")
    print(f"  Test:  {len(i_test)} samples")
    
    # Create datasets
    train_dataset = EnhancedMyanmarDataset(
        CSV_PATH, H5_PATH, indices=i_train, augment=True
    )
    val_dataset = EnhancedMyanmarDataset(
        CSV_PATH, H5_PATH, indices=i_val, augment=False
    )
    test_dataset = EnhancedMyanmarDataset(
        CSV_PATH, H5_PATH, indices=i_test, augment=False
    )
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyanmarCNNEnhanced(num_classes=2).to(device)
    
    # Load pretrained weights if available
    pretrained_path = r"C:\Users\USER\eew\models\cnn_baseline\cnn_baseline.pt"
    if os.path.exists(pretrained_path):
        print(f"\nLoading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Try to load partial weights
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
    
    # Focal loss for imbalanced data
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Optimizer with different learning rates for different layers
    optimizer = optim.AdamW([
        {'params': model.conv1.parameters(), 'lr': 1e-5},
        {'params': model.conv2.parameters(), 'lr': 1e-5},
        {'params': model.conv3.parameters(), 'lr': 1e-4},
        {'params': model.conv4.parameters(), 'lr': 1e-4},
        {'params': model.attention.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Training
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\n" + "="*70)
    print("STARTING ENHANCED TRAINING")
    print("="*70)
    
    for epoch in range(30):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/30")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': train_correct / train_total
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).squeeze()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss / train_total)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / val_total)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train - Loss: {train_loss/train_total:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss/val_total:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, os.path.join(OUTPUT_DIR, "best_model.pt"))
            
            print(f"  → Saved best model (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  → No improvement for {patience_counter}/{patience} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        # Update scheduler
        scheduler.step()
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device).squeeze()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='binary')
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1 Score: {test_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Noise', 'Earthquake']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("           Predicted")
    print("           Noise  EQ")
    print(f"Actual Noise  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"        EQ     {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'best_val_accuracy': float(best_val_acc),
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(OUTPUT_DIR, "final_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History - Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best Val: {best_val_acc:.3f}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training History - Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=300)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    if test_acc >= 0.90:
        print(f"🎉 SUCCESS! Achieved >90% accuracy: {test_acc:.4f}")
    else:
        print(f"📈 Good progress! Current accuracy: {test_acc:.4f}")
        print("   Consider: Download more noise data (run download script)")
    
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print(f"Results saved to: {os.path.join(OUTPUT_DIR, 'final_results.json')}")

if __name__ == "__main__":

    main()
