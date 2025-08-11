import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm
import wandb
from src.data.dataset import CustomDataset
from src.models.base_model import AnomalyModel
from src.utils.metrics import multi_class_accuracy, evaluate_model

def train():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    wandb.init(project="anomaly-detection")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = CustomDataset(config['data']['train_path'], train=True)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    model = AnomalyModel(config['model']['num_classes']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['model']['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for seqs, labels in progress:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(seqs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        acc = multi_class_accuracy(model, train_loader, device)
        wandb.log({'epoch': epoch, 'loss': avg_loss, 'acc': acc, 'lr': scheduler.get_last_lr()[0]})
        scheduler.step()
        
        if acc > 0.95:
            torch.save(model.state_dict(), 'models_saved/best_model.pth')
            print("Achieved >95% accuracy!")
    
    # Evaluation
    test_dataset = CustomDataset('data/processed/test', train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    classes = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Normal Videos', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
    evaluate_model(model, test_loader, device, classes)
    
    return model, device

if __name__ == '__main__':
    train()