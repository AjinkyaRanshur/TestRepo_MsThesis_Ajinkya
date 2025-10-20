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

        # Create a mapping of class names to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes_for_use)}
        
        # Map class names to their corresponding indices
        self.metadata['Label'] = self.metadata['Class'].apply(
            lambda x: self.class_to_idx[x])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.metadata.iloc[idx]['Class'],
                                self.metadata.iloc[idx]['filename'])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(
            self.metadata.iloc[idx]['Label'],
            dtype=torch.long)
        cls_name = self.metadata.iloc[idx]['Class']
        return image, label, cls_name
