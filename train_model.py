# train_model.py
import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from training import AudioDataset, CRNN, evaluate_model, train_model, CHECKPOINT_DIR, DATASET_PATHS

def start_training():
    dataset_path = DATASET_PATHS[0]  # You can loop or combine multiple datasets as needed
    full_dataset = AudioDataset(dataset_path)
    indices = list(range(len(full_dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=full_dataset.labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=full_dataset.labels[train_idx])

    train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_idx), batch_size=16, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(full_dataset, val_idx), batch_size=32)
    test_loader = DataLoader(torch.utils.data.Subset(full_dataset, test_idx), batch_size=32)

    model = CRNN()
    train_model(model, train_loader, val_loader, epochs=20)
    print("Final Test Evaluation:")
    evaluate_model(model, test_loader)
