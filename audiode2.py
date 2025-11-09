
# training.py
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATASET_PATHS = ['./audio1', './Dataset_B']  # Add your dataset folders here
SAMPLE_RATE = 16000
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) < SAMPLES_PER_TRACK:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
    else:
        y = y[:SAMPLES_PER_TRACK]
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9)
    return mel_spec_db

class AudioDataset(Dataset):
    def __init__(self, base_path):
        self.data = []
        self.labels = []

        real_folder = os.path.join(base_path, "Real")
        synth_folder = os.path.join(base_path, "Synthesized")
        real_files = [f for f in os.listdir(real_folder) if f.endswith(".wav")]
        synth_files = [f for f in os.listdir(synth_folder) if f.endswith(".wav")]
        max_len = max(len(real_files), len(synth_files))

        real_files = (real_files * (max_len // len(real_files) + 1))[:max_len]
        synth_files = (synth_files * (max_len // len(synth_files) + 1))[:max_len]

        for filename in real_files:
            full_path = os.path.join(real_folder, filename)
            features = preprocess_audio(full_path)
            self.data.append(features)
            self.labels.append(0)

        for filename in synth_files:
            full_path = os.path.join(synth_folder, filename)
            features = preprocess_audio(full_path)
            self.data.append(features)
            self.labels.append(1)

        self.data = torch.tensor(np.array(self.data)).unsqueeze(1).float()
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.rnn = nn.GRU(64 * 16, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2)
        B, T, C, F = x.shape
        x = x.reshape(B, T, C * F)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

def evaluate_model(model, dataloader, print_report=True):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            output = model(X)
            preds.extend(output.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    acc = accuracy_score(labels, preds)
    if print_report:
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(labels, preds, target_names=["Real", "Fake"]))
    return acc

def train_and_evaluate_on_dataset(dataset_path, model_name="model", epochs=20):
    print(f"\n Processing dataset: {dataset_path}")
    dataset = AudioDataset(dataset_path)
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=dataset.labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=dataset.labels[train_idx])

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=16, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=32, pin_memory=pin_memory)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=32, pin_memory=pin_memory)

    model = CRNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_best_model.pth')
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == y).sum().item()
        train_acc = correct / len(train_loader.dataset)

        val_acc = evaluate_model(model, val_loader, print_report=False)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
        print(f"[{model_name}] Epoch {epoch+1}/{epochs} - Train Loss: {running_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    print(f"\n Final Evaluation for {model_name}:")
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    for i, path in enumerate(DATASET_PATHS):
        train_and_evaluate_on_dataset(path, model_name=f"model_{i+1}", epochs=20)
