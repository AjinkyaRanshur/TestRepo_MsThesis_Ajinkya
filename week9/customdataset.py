import pandas as pd
import os
import torch  # FIXED: Added missing import
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset  # FIXED: Added missing import


class SquareDataset(Dataset):
    def __init__(self, csv_file, img_dir, classes_for_use, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.metadata = self.metadata[self.metadata['Class'].isin(
            classes_for_use)]
        self.img_dir = img_dir
        self.transform = transform
        self.metadata['Label'] = self.metadata['Class'].apply(
            lambda x: 1 if x == "Square" else 0)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.metadata.iloc[idx]['Class'],
                                self.metadata.iloc[idx]['filename'])
        image = Image.open(img_name).convert("L")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(
            self.metadata.iloc[idx]['Label'],
            dtype=torch.float32)
        cls_name = self.metadata.iloc[idx]['Class']
        return image, label, cls_name
