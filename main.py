import torchaudio
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F


class UrbanSoundDataset(Dataset):
    def __init__(self, root_dir, subset=None):
        self.dataset = URBANSOUND8K(root=root_dir, download=True, subset=subset)
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_mels=64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, label, _, _, _ = self.dataset[idx]
        mel_spec = self.transform(waveform)
        return mel_spec, label


class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 16 * 32, 128)
        self.fc2 = nn.Linear(128, 10)  # 假设有10个类别

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




if __name__ == '__main__':
    # 加载数据和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassifier().to(device)
    train_dataset = UrbanSoundDataset(root_dir='./data', subset='training')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # # 训练模型
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # for epoch in range(10):
    #     for mel_specs, labels in train_loader:
    #         mel_specs, labels = mel_specs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(mel_specs)
    #         loss = F.cross_entropy(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #     print(f'Epoch {epoch + 1}, Loss: {loss.item()}')