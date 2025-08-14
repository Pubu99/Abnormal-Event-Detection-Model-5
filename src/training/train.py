import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import yaml
from tqdm import tqdm
import wandb
from collections import Counter
import sys
import os
import numpy as np
import time
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from src.data.dataset import CustomDataset
from src.models.base_model import AnomalyModel
from src.utils.metrics import multi_class_accuracy, evaluate_model

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=3, reduction='mean'):  # Increased gamma to 3
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

def compute_val_loss(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        progress = tqdm(val_loader, desc="Validation", leave=True, file=sys.stdout)
        for batch_idx, (seqs, labels, yolo_features) in enumerate(progress):
            seqs, labels, yolo_features = seqs.to(device), labels.to(device), yolo_features.to(device)
            with autocast('cuda'):
                outputs = model(seqs, yolo_features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            progress.set_postfix(loss=loss.item())
    return val_loss / len(val_loader)

def train():
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    wandb.init(project="anomaly-detection", config=config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading training dataset...")
    full_dataset = CustomDataset(config['data']['train_path'], config['data']['train_yolo_path'], train=True)
    print(f"Dataset size: {len(full_dataset)} samples")
    val_ratio = config['training'].get('val_ratio', 0.1)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Validation size: {val_size}")

    print("Computing class weights for oversampling...")
    label_counts_path = os.path.join(config['data']['train_path'], 'label_counts.yaml')
    with open(label_counts_path, 'r') as f:
        counts = yaml.safe_load(f)
    print("Class counts from label_counts.yaml:", counts)
    
    train_labels = train_dataset.dataset.labels
    train_indices = train_dataset.indices
    train_labels = [train_labels[i] for i in train_indices]
    
    label_dist = Counter(train_labels)
    print("Training label distribution:", {i: label_dist.get(i, 0) for i in range(config['model']['num_classes'])})
    
    class_sample_counts = np.array([label_dist.get(i, 1) for i in range(config['model']['num_classes'])])
    weights_per_class = 1.0 / (class_sample_counts ** 0.7)  # Adjusted for stronger minority class sampling
    class_weights = [weights_per_class[label] for label in train_labels]
    print(f"Class weights (first 5 samples): {class_weights[:5]}")
    
    sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(train_labels), replacement=True)
    print("WeightedRandomSampler initialized.")
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print("Initializing model...")
    model = AnomalyModel(config['model']['num_classes']).to(device)
    print(f"GPU Memory after model init: {torch.cuda.memory_allocated(device)/1e9:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved(device)/1e9:.2f} GB reserved")
    
    num_classes = config['model']['num_classes']
    alpha = torch.tensor([1.0 / max(counts.get(i, 1), 1) for i in range(num_classes)], device=device)
    alpha = torch.clamp(alpha, min=1e-3)
    alpha = alpha / alpha.sum() * num_classes
    print(f"Focal Loss alpha: {alpha.cpu().numpy().tolist()}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config['model']['lr'], weight_decay=0.05)  # Increased weight decay
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['model']['lr'], 
        steps_per_epoch=len(train_loader), epochs=config['training']['epochs']
    )
    criterion = FocalLoss(alpha=alpha, gamma=3)  # Updated gamma
    
    scaler = GradScaler('cuda')
    
    checkpoint_dir = 'models_saved'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    start_epoch = 0
    best_macro_f1 = 0.0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_macro_f1 = checkpoint['best_macro_f1']
        print(f"Resumed from checkpoint at epoch {start_epoch} with best macro F1={best_macro_f1:.4f}")
    
    patience = config['training'].get('patience', 5)
    counter = 0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        epoch_loss = 0
        yolo_time = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", 
                        leave=True, file=sys.stdout)
        for batch_idx, (seqs, labels, yolo_features) in enumerate(progress):
            seqs, labels, yolo_features = seqs.to(device), labels.to(device), yolo_features.to(device)
            optimizer.zero_grad()
            start_time = time.time()
            with autocast('cuda'):
                outputs = model(seqs, yolo_features)
                loss = criterion(outputs, labels)
            yolo_time += time.time() - start_time
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item(), scaler=scaler.get_scale())
        
        avg_loss = epoch_loss / len(train_loader)
        train_acc = multi_class_accuracy(model, train_loader, device)
        
        val_loss = compute_val_loss(model, val_loader, criterion, device)
        val_acc = multi_class_accuracy(model, val_loader, device)
        
        val_preds, val_labels = [], []
        with torch.no_grad():
            for seqs, labels, yolo_features in val_loader:
                seqs, labels, yolo_features = seqs.to(device), labels.to(device), yolo_features.to(device)
                with autocast('cuda'):
                    outputs = model(seqs, yolo_features)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        val_macro_f1 = precision_recall_fscore_support(val_labels, val_preds, average='macro')[2]
        
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_macro_f1': val_macro_f1,
            'lr': scheduler.get_last_lr()[0],
            'scaler': scaler.get_scale(),
            'yolo_time': yolo_time,
            'gpu_alloc_gb': torch.cuda.memory_allocated(device)/1e9,
            'gpu_reserved_gb': torch.cuda.memory_reserved(device)/1e9
        })
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation Macro F1: {val_macro_f1:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f} GB allocated, "
              f"{torch.cuda.memory_reserved(device)/1e9:.2f} GB reserved")
        print(f"  Scaler: {scaler.get_scale():.2f}")
        print(f"  YOLO Time: {yolo_time:.2f}s")
        
        if epoch % 5 == 0 or epoch == config['training']['epochs'] - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_macro_f1': best_macro_f1
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
        
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"New best model saved at epoch {epoch+1} with val_macro_f1={val_macro_f1:.4f}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    print("Loading test dataset...")
    test_dataset = CustomDataset(config['data']['test_path'], config['data']['test_yolo_path'], train=False)
    print(f"Test dataset size: {len(test_dataset)} samples")
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    classes = [
        'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
        'Fighting', 'Normal Videos', 'RoadAccidents', 'Robbery',
        'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
    ]
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth')))
    print("Evaluating on test set...")
    acc, macro_f1, weighted_f1 = evaluate_model(model, test_loader, device, classes)
    print(f"Test Evaluation Complete: Accuracy={acc:.4f}, Macro F1={macro_f1:.4f}, Weighted F1={weighted_f1:.4f}")
    
    eval_dir = os.path.join(checkpoint_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    txt_path = os.path.join(eval_dir, 'test_metrics.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Test Evaluation Complete: Accuracy={acc:.4f}, Macro F1={macro_f1:.4f}, Weighted F1={weighted_f1:.4f}\n")
        f.write("\nPer-Class Metrics:\n")
        f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
        f.write("-" * 60 + "\n")
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seqs, labels, yolo_features in test_loader:
                seqs, labels, yolo_features = seqs.to(device), labels.to(device), yolo_features.to(device)
                with autocast('cuda'):
                    outputs = model(seqs, yolo_features)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=range(len(classes)))
        for i, cls in enumerate(classes):
            f.write(f"{cls:<20} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}\n")
        f.write(f"\nMacro Average F1: {macro_f1:.4f}\n")
        f.write(f"Weighted Average F1: {weighted_f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        cm = confusion_matrix(all_labels, all_preds)
        f.write(str(cm) + "\n")
    
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    wandb.log({"confusion_matrix": wandb.Table(dataframe=cm_df)})
    
    print(f"Test metrics saved to {txt_path}")
    
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    cm_path = os.path.join(eval_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    return model, device

def multi_class_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        progress = tqdm(loader, desc="Computing Accuracy", leave=True, file=sys.stdout)
        for batch_idx, (seqs, labels, yolo_features) in enumerate(progress):
            seqs, labels, yolo_features = seqs.to(device), labels.to(device), yolo_features.to(device)
            with autocast('cuda'):
                outputs = model(seqs, yolo_features)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(batch_accuracy=correct/total)
    return correct / total

if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    train()