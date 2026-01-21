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
        row = self.metadata.iloc[idx]

        # Full image path
        img_path = os.path.join(self.img_dir, row["Class"], row["filename"])

        # Load as RGB (still grayscale if original images are gray)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["Label"], dtype=torch.long)
        cls_name = row["Class"]

        # NEW: load illusion ground-truth perception
        should_see = row["Should_See"]

        return image, label, cls_name, should_see




