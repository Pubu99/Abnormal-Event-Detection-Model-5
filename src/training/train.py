import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
import yaml
from tqdm import tqdm
import wandb
from collections import Counter
import sys
import os
from src.data.dataset import CustomDataset
from src.models.base_model import AnomalyModel
from src.utils.metrics import multi_class_accuracy, evaluate_model

def compute_val_loss(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        progress = tqdm(val_loader, desc="Validation", leave=True, file=sys.stdout)
        for batch_idx, (seqs, labels) in enumerate(progress):
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            progress.set_postfix(loss=loss.item())
    return val_loss / len(val_loader)

def train():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    wandb.init(project="anomaly-detection", config=config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved(device)/1e9:.2f} GB reserved")
    
    # --- Dataset split into train/val ---
    print("Loading training dataset...")
    full_dataset = CustomDataset(config['data']['train_path'], train=True)
    print(f"Dataset size: {len(full_dataset)} samples")
    val_ratio = config['training'].get('val_ratio', 0.1)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Validation size: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=2, pin_memory=True)
    
    # --- Model, optimizer, scheduler ---
    print("Initializing model...")
    model = AnomalyModel(config['model']['num_classes']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['model']['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    # --- Class weights for imbalance ---
    print("Loading class weights...")
    label_counts_path = os.path.join(config['data']['train_path'], 'label_counts.yaml')
    if not os.path.exists(label_counts_path):
        raise FileNotFoundError(f"Label counts file not found at {label_counts_path}. Run preprocess.py first.")
    with open(label_counts_path, 'r') as f:
        counts = Counter(yaml.safe_load(f))
    num_classes = config['model']['num_classes']
    weights = []
    for i in range(num_classes):
        weight = 1.0 / counts.get(i, 1e-6)
        weights.append(weight)
    weights = torch.tensor(weights).to(device)
    weights = weights / weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"Class weights: {weights.tolist()}")
    
    scaler = GradScaler('cuda')
    
    # --- Resume from checkpoint ---
    checkpoint_dir = 'models_saved'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    start_epoch = 0
    best_acc = 0.0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resumed from checkpoint at epoch {start_epoch} with best val_acc={best_acc:.4f}")
    
    # --- Early stopping variables ---
    patience = config['training'].get('patience', 5)
    counter = 0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        epoch_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", 
                        leave=True, file=sys.stdout)
        for batch_idx, (seqs, labels) in enumerate(progress):
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(seqs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        train_acc = multi_class_accuracy(model, train_loader, device)
        
        # --- Validation ---
        val_loss = compute_val_loss(model, val_loader, criterion, device)
        val_acc = multi_class_accuracy(model, val_loader, device)
        
        # --- Logging ---
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # --- Epoch summary ---
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f} GB allocated, "
              f"{torch.cuda.memory_reserved(device)/1e9:.2f} GB reserved")
        
        # --- Save checkpoint ---
        if epoch % 5 == 0 or epoch == config['training']['epochs'] - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
        
        # --- Early stopping check ---
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"New best model saved at epoch {epoch+1} with val_acc={val_acc:.4f}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    # --- Evaluation on test set ---
    print("Loading test dataset...")
    test_dataset = CustomDataset(config['data']['test_path'], train=False)
    print(f"Test dataset size: {len(test_dataset)} samples")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    classes = [
        'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
        'Fighting', 'Normal Videos', 'RoadAccidents', 'Robbery',
        'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
    ]
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth')))
    print("Evaluating on test set...")
    acc, macro_f1, weighted_f1 = evaluate_model(model, test_loader, device, classes)
    print(f"Test Evaluation Complete: Accuracy={acc:.4f}, Macro F1={macro_f1:.4f}, Weighted F1={weighted_f1:.4f}")
    
    return model, device

if __name__ == "__main__":
    import os
    os.environ["PYTHONUNBUFFERED"] = "1"
    train()