import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


PATCH_SIZE = 64

class JPEGDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.image_list = self._get_image_list()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _get_image_list(self):
        image_list = []

        for subdir, _, files in os.walk(self.root_path):
            for file in files:
                if file.lower().endswith(".jpg"):
                    image_list.append(os.path.join(subdir, file))

        return image_list
    


    def _extract_all_patches(self, image, patch_size=PATCH_SIZE):
        image = np.array(image)
        image_height, image_width, _ = image.shape
        patches = []

        num_patches_height = image_height // patch_size
        num_patches_width = image_width // patch_size

        for i in range(num_patches_height):
            for j in range(num_patches_width):
                start_x = j * patch_size
                start_y = i * patch_size
                patch = image[start_y:start_y + patch_size, start_x:start_x + patch_size, :]
                patches.append(patch)

        return patches
    

    def _extract_random_patches(self, image, patch_size=PATCH_SIZE, num_patches=2):
        image = np.array(image)
        image_height, image_width, _ = image.shape
        patches = []


        for _ in range(num_patches):
            start_x = np.min((np.random.randint(1, image_width // 8) * 8, image_width - patch_size)) - 1
            start_y = np.min((np.random.randint(1, image_height // 8) * 8, image_height - patch_size)) - 1

            patch = image[start_y:start_y + patch_size, start_x:start_x + patch_size, :]
            patches.append(patch)


        return patches
    
    def get_all_patches(self, idx):
        image = Image.open(self.image_list[idx]).convert('RGB')
        
        patches = np.stack(self._extract_all_patches(image), axis=0)

        return patches

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image = Image.open(self.image_list[idx]).convert('RGB')

        patches = np.stack(self._extract_random_patches(image), axis=0)

        patches = torch.stack((self.transform(patches[0]), self.transform(patches[1])))

        return patches

def create_dataloader(dataset, batch_size = 32):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


class PNGDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.image_list = self._get_image_list()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _get_image_list(self):
        image_list = []

        for subdir, _, files in os.walk(self.root_path):
            for file in files:
                if file.lower().endswith(".png"):
                    image_list.append(os.path.join(subdir, file))

        return image_list
    


    def _extract_all_patches(self, image, patch_size=PATCH_SIZE):
        image = np.array(image)
        image_height, image_width, _ = image.shape
        patches = []

        num_patches_height = image_height // patch_size
        num_patches_width = image_width // patch_size

        for i in range(num_patches_height):
            for j in range(num_patches_width):
                start_x = j * patch_size
                start_y = i * patch_size
                patch = image[start_y:start_y + patch_size, start_x:start_x + patch_size, :]
                patches.append(patch)

        return patches
    

    def _extract_random_patches(self, image, patch_size=PATCH_SIZE, num_patches=2):
        image = np.array(image)
        image_height, image_width, _ = image.shape
        patches = []


        for _ in range(num_patches):
            start_x = np.min((np.random.randint(1, image_width // 8) * 8, image_width - patch_size)) - 1
            start_y = np.min((np.random.randint(1, image_height // 8) * 8, image_height - patch_size)) - 1

            patch = image[start_y:start_y + patch_size, start_x:start_x + patch_size, :]
            patches.append(patch)


        return patches
    
    def get_all_patches(self, idx):
        image = Image.open(self.image_list[idx]).convert('RGB')
        
        patches = np.stack(self._extract_all_patches(image), axis=0)

        return patches

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image = Image.open(self.image_list[idx]).convert('RGB')

        patches = np.stack(self._extract_random_patches(image), axis=0)

        patches = torch.stack((self.transform(patches[0]), self.transform(patches[1])))

        return patches

def create_dataloader(dataset, batch_size = 32):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def calculate_entropy(patch):
    histogram, _ = np.histogram(patch, bins=256, range=(0, 256), density=True)
    histogram = histogram[np.nonzero(histogram)]
    return -np.sum(histogram * np.log2(histogram))

def select_informative_patches(patches, num_patches=None):
    entropies = np.array([calculate_entropy(patch) for patch in patches])
    if num_patches:
        informative_indices = np.argsort(entropies)[-num_patches:]
    else:
        informative_indices = np.argsort(entropies)
    return patches[informative_indices]