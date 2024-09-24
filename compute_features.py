import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from tqdm import tqdm
import os
from data_utils import PNGDataset, JPEGDataset, select_informative_patches
from pretrain_real_ae import ConvAutoencoder


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

def calculate_rec_errors(patches, encoders, decoder, device):
    patches = patches.to(device)
    patch_errors = np.zeros((len(encoders), patches.size(0)))

    criterion = nn.MSELoss()

    for i, encoder in enumerate(encoders):
        encoder.to(device)
        encoder.eval()
        with torch.no_grad():
            encoded = encoder(patches)
            reconstructions = decoder(encoded)
            for j in range(patches.size(0)):
                patch_errors[i, j] = criterion(reconstructions[j], patches[j]).item()
    return patch_errors



if __name__ == "__main__":
    for data_split in ['Train', 'Test']:
        for seed in [41, 51, 61, 71, 81]:
            for n_train in [5, 10, 20]:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print("Using device:", device)

                autoencoder = ConvAutoencoder().to(device)
                autoencoder.load_state_dict(torch.load('model_weights/convAE.pth'))
                decoder = autoencoder.decoder
                decoder.eval()

                datasets_path = "GenImage"
                encoders = []
                dataset_names = []
                test_images = []
                ground_truths = []

                list_dir = os.listdir(datasets_path)

                for dataset_name in list_dir:
                    dataset_path = os.path.join(datasets_path, dataset_name)
                    dataset_names.append(dataset_name)
                    if dataset_name == 'real':
                        ds = JPEGDataset(root_path=dataset_path)
                    else:
                        ds = PNGDataset(root_path=dataset_path)
                    
                    train_size = n_train
                    test_size = len(ds) - train_size
                    if data_split == 'Test':
                        train_indices, test_indices = random_split(range(len(ds)), [train_size, test_size])
                    else:
                        test_indices, train_indices = random_split(range(len(ds)), [train_size, test_size])

                    
                    encoder = Encoder()
                    encoder_path = f'model_weights/{dataset_name}_encoder_seed_{seed}_train_size_{train_size}.pth'

                    print(f'Loading pre-trained {dataset_name} encoder')
                    encoder.load_state_dict(torch.load(encoder_path))
                    encoders.append(encoder)

                    for idx in tqdm(test_indices):
                        all_patches = ds.get_all_patches(idx)
                        informative_patches = select_informative_patches(all_patches, num_patches=16)
                        test_images.append(informative_patches)
                        ground_truths.append(dataset_name)

                rec_errors = {label: [] for label in dataset_names}
                rec_errors_per_patch = {label: [] for label in dataset_names}

                for img_patches, label in zip(test_images, ground_truths):
                    img_patches = np.transpose(img_patches, (0, 3, 1, 2))
                    img_patches = torch.Tensor(img_patches).to(device)
                    rec_error_per_patch = calculate_rec_errors(img_patches, encoders, decoder, device)
                    rec_errors_per_patch[label].append(np.array(rec_error_per_patch))

                for label in rec_errors_per_patch.keys():
                    r = np.array(rec_errors_per_patch[label])
                    for i in range(len(encoders)):
                        save_path = f'rec_errors/{data_split}/{seed}/{n_train}'
                        os.makedirs(save_path, exist_ok=True)
                        np.save(f'{save_path}/rec_errors_encoder_{dataset_names[i]}_dataset_{label}.npy', r[:, i, :])