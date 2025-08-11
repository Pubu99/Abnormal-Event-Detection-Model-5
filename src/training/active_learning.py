import torch
import numpy as np
from sklearn.metrics.pairwise import entropy
from torch.utils.data import DataLoader, TensorDataset

def active_learning_loop(model, unlabeled_loader: DataLoader, device: torch.device, num_queries: int = 100):
    model.eval()
    uncertainties = []
    samples = []
    for seqs in unlabeled_loader:
        seqs = seqs.to(device)  # Assume unlabeled has only seqs
        with torch.no_grad():
            scores = model(seqs)
        unc = entropy(scores.cpu().numpy(), axis=1)
        uncertainties.extend(unc)
        samples.extend(seqs.cpu())
    
    top_idx = np.argsort(uncertainties)[-num_queries:]
    query_samples = [samples[i] for i in top_idx]
    
    # User feedback: In practice, send to app/API for labeling
    new_labels = []  # Placeholder: assume user provides list of labels
    # For demo, simulate or integrate with backend
    
    fine_tune_dataset = TensorDataset(torch.stack(query_samples), torch.tensor(new_labels))
    fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=32, shuffle=True)
    
    # Fine-tune (similar to train loop)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(5):  # Short fine-tune
        for seqs, labels in fine_tune_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()