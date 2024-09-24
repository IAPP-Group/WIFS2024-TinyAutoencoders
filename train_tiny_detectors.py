import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os
from data_utils import PNGDataset, JPEGDataset, select_informative_patches
from pretrain_real_ae import ConvAutoencoder


num_epochs = 2000


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        return self.data_tensor[idx]
    

def train_encoder(encoder, decoder, dataloader, device, num_epochs, lr=0.001):
    encoder.to(device)
    decoder.to(device)
    decoder.eval()

    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    last_loss = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f'Epoch [{epoch+1}/{num_epochs}]'):
            batch = batch.to(device)
            encoded = encoder(batch)
            reconstructions = decoder(encoded)
            loss = criterion(reconstructions, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        last_loss = epoch_loss
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return encoder, last_loss


if __name__ == "__main__":

    for seed in [41, 51, 61, 71, 81]:
        for n_train in [5, 10, 20]:
            set_seed(seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Using device:", device)
            autoencoder = ConvAutoencoder().to(device)
            autoencoder.load_state_dict(torch.load('model_weights/convAE.pth'))
            decoder = autoencoder.decoder
            decoder.eval()

            datasets_path = "GenImage"
            encoders = []
            dataset_names = []

            list_dir = os.listdir(datasets_path)
            print(list_dir)

            for dataset_name in list_dir:
                dataset_path = os.path.join(datasets_path, dataset_name)
                dataset_names.append(dataset_name)
                if dataset_name == 'real':
                    ds = JPEGDataset(root_path=dataset_path)
                else:
                    ds = PNGDataset(root_path=dataset_path)
                
                train_size = n_train
                test_size = len(ds) - train_size
                train_indices, test_indices = random_split(range(len(ds)), [train_size, test_size])
                print(f'Train size = {train_size} --- Test size = {test_size}')

                train_patches = []
                for idx in train_indices:
                    all_patches = ds.get_all_patches(idx)
                    informative_patches = select_informative_patches(all_patches, num_patches=16)
                    train_patches.append(informative_patches)

                train_patches = np.concatenate(train_patches)
                train_patches = np.transpose(train_patches, (0, 3, 1, 2))
                train_patches = torch.Tensor(train_patches)
                train_dl = DataLoader(CustomDataset(train_patches), batch_size=train_size, shuffle=True)

                encoder = Encoder()
                encoder_path = f'model_weights/{dataset_name}_encoder_seed_{seed}_train_size_{train_size}.pth'
                
                print(f'Training {dataset_name} encoder')
                trained_encoder, last_loss = train_encoder(encoder, decoder, train_dl, device, num_epochs)
                torch.save(trained_encoder.state_dict(), encoder_path)
