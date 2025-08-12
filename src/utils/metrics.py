import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import numpy as np
from tqdm import tqdm
import sys

def evaluate_model(model, test_loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress = tqdm(test_loader, desc="Evaluating Test Set", leave=True, file=sys.stdout)
        for batch_idx, (seqs, labels) in enumerate(progress):
            seqs, labels = seqs.to(device), labels.to(device)
            print(f"Test Batch {batch_idx+1}/{len(test_loader)}: Input Shape={seqs.shape}")
            outputs = model(seqs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            progress.set_postfix(batch_accuracy=accuracy_score(labels.cpu().numpy(), preds))
            print(f"Test Batch {batch_idx+1}/{len(test_loader)}: Batch Accuracy={accuracy_score(labels.cpu().numpy(), preds):.4f}")
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=range(len(classes)))
    macro_f1 = np.mean(f1)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\nOverall Accuracy: {acc:.4f}")
    for i, cls in enumerate(classes):
        print(f"{cls}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Confusion Matrix:\n", cm)
    
    return acc, macro_f1

def multi_class_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        progress = tqdm(loader, desc="Computing Accuracy", leave=True, file=sys.stdout)
        for batch_idx, (seqs, labels) in enumerate(progress):
            seqs, labels = seqs.to(device), labels.to(device)
            print(f"Accuracy Batch {batch_idx+1}/{len(loader)}: Input Shape={seqs.shape}")
            outputs = model(seqs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(batch_accuracy=correct/total)
            print(f"Accuracy Batch {batch_idx+1}/{len(loader)}: Batch Accuracy={correct/total:.4f}")
    return correct / total