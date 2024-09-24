import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_utils import JPEGDataset

class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        return self.data_tensor[idx]


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds = JPEGDataset(root_path='flickr30k_images')

    print(len(ds))

    patches = []
    for idx in tqdm(range(len(ds))):
        patches.append(ds.get_all_patches(idx))

    

    all_patches = np.concatenate([image_patches for image_patches in patches])
    all_patches = np.transpose(all_patches, (0, 3, 1, 2))
    print(all_patches.shape)

    all_patches = torch.Tensor(all_patches)
    print(all_patches.dtype)

    dataset = CustomDataset(all_patches)

    dl = DataLoader(dataset, batch_size=512)


    autoencoder = ConvAutoencoder().to(device)

    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 1000
    autoencoder.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(dl):
            batch = batch.to(device)
            reconstructions = autoencoder(batch)
            loss = criterion(reconstructions, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dl)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    os.makedirs('model_weights', exist_ok=True)
    torch.save(autoencoder.state_dict(), 'model_weights/convAE.pth')