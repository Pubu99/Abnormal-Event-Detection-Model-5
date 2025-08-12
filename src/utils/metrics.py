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
            outputs = model(seqs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            progress.set_postfix(batch_accuracy=accuracy_score(labels.cpu().numpy(), preds))
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, 
                                                                    average=None, labels=range(len(classes)))
    macro_f1 = np.mean(f1)
    weighted_f1 = np.sum(f1 * support) / np.sum(support)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print detailed evaluation
    print("\nTest Set Evaluation:")
    print(f"Overall Accuracy: {acc:.4f}")
    print("\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)
    for i, cls in enumerate(classes):
        print(f"{cls:<20} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
    print(f"\nMacro Average F1: {macro_f1:.4f}")
    print(f"Weighted Average F1: {weighted_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    return acc, macro_f1, weighted_f1

def multi_class_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        progress = tqdm(loader, desc="Computing Accuracy", leave=True, file=sys.stdout)
        for batch_idx, (seqs, labels) in enumerate(progress):
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(batch_accuracy=correct/total)
    return correct / total