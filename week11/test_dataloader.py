"""
Test script for illusion dataset loading
Ensures proper class balance across train/validation/test splits
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from collections import Counter


class SquareDataset(Dataset):
    def __init__(self, csv_file, img_dir, classes_for_use, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.metadata = self.metadata[self.metadata['Class'].isin(classes_for_use)]
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes_for_use)}
        self.metadata['Label'] = self.metadata['Class'].apply(lambda x: self.class_to_idx[x])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Class"], row["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row["Label"], dtype=torch.long)
        cls_name = row["Class"]
        should_see = row["Should_See"]
        return image, label, cls_name, should_see


def analyze_distribution(indices, dataset, split_name):
    """Analyze class distribution for a set of indices"""
    print(f"\n{'='*70}")
    print(f"{split_name.upper()} SET ANALYSIS")
    print(f"{'='*70}")
    
    class_counts = {}
    should_see_counts = {}
    
    for idx in indices:
        _, _, cls_name, should_see = dataset[idx]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        # Track should_see distribution
        key = f"{cls_name} → {should_see}"
        should_see_counts[key] = should_see_counts.get(key, 0) + 1
    
    print(f"\nTotal samples: {len(indices)}")
    print(f"\nClass distribution:")
    for cls_name in sorted(class_counts.keys()):
        print(f"  {cls_name:15s}: {class_counts[cls_name]:5d} samples")
    
    # Show should_see breakdown for random, all_in, all_out
    print(f"\n'Should_See' breakdown:")
    for key in sorted(should_see_counts.keys()):
        print(f"  {key:30s}: {should_see_counts[key]:5d} samples")
    
    return class_counts, should_see_counts


def create_balanced_dataloaders(data_dir, batch_size=40, seed=42):
    """
    Create train/val/test dataloaders with proper class balance
    
    Rules:
    1. Basic shapes (square, hexagon, triangle, trapezium, rectangle) have k samples each
    2. Random has 5k samples
    3. Train: 70% of all basic + random, stratified by should_see for random
    4. Val: 30% of all basic + random, stratified by should_see for random
    5. Test: Val set + equal samples from all_in/all_out (matching random count in val)

    """
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    ALL_CLASSES = ["square", "rectangle", "trapezium", "triangle", "hexagon", 
                   "random", "all_in", "all_out"]
    BASIC_CLASSES = ["square", "rectangle", "trapezium", "triangle", "hexagon", "random"]
    
    # Load full dataset
    full_dataset = SquareDataset(
        os.path.join(data_dir, "dataset_metadata.csv"),
        data_dir,
        classes_for_use=ALL_CLASSES,
        transform=transform
    )
    
    print(f"\n{'='*70}")
    print(f"DATASET LOADING")
    print(f"{'='*70}")
    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Class mapping: {full_dataset.class_to_idx}")
    
    # ================================================================
    # STEP 1: Separate indices by class
    # ================================================================
    basic_shape_indices = {cls: [] for cls in ["square", "rectangle", "trapezium", 
                                                "triangle", "hexagon"]}
    random_indices_by_should_see = {}
    illusion_indices = {"all_in": [], "all_out": []}
    
    for i in range(len(full_dataset)):
        _, _, cls_name, should_see = full_dataset[i]
        
        if cls_name in basic_shape_indices:
            basic_shape_indices[cls_name].append(i)
        elif cls_name == "random":
            if should_see not in random_indices_by_should_see:
                random_indices_by_should_see[should_see] = []
            random_indices_by_should_see[should_see].append(i)
        elif cls_name in illusion_indices:
            illusion_indices[cls_name].append(i)
    
    # Print initial counts
    print(f"\nInitial class distribution:")
    for cls in basic_shape_indices:
        print(f"  {cls:15s}: {len(basic_shape_indices[cls])} samples")
    print(f"  {'random':15s}: {sum(len(v) for v in random_indices_by_should_see.values())} samples")
    for see_cls, indices in random_indices_by_should_see.items():
        print(f"    → should_see '{see_cls}': {len(indices)}")
    for cls in illusion_indices:
        print(f"  {cls:15s}: {len(illusion_indices[cls])} samples")
    
    # ================================================================
    # STEP 2: Split basic shapes 70-30
    # ================================================================
    train_indices = []
    val_indices = []
    
    for cls_name, indices in basic_shape_indices.items():
        indices = np.array(indices)
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.3,
            random_state=seed
        )
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
    
    # ================================================================
    # STEP 3: Split random 70-30 STRATIFIED by should_see
    # ================================================================
    for should_see_cls, indices in random_indices_by_should_see.items():
        indices = np.array(indices)
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.3,
            random_state=seed
        )
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    
    # Analyze distributions
    train_class_counts, train_should_see = analyze_distribution(
        train_indices, full_dataset, "TRAINING"
    )
    val_class_counts, val_should_see = analyze_distribution(
        val_indices, full_dataset, "VALIDATION"
    )
    
    # ================================================================
    # STEP 4: Sample all_in and all_out for test set
    # ================================================================
    # Number to sample = total random in validation
    random_count_in_val = val_class_counts.get('random', 0)
    
    # Calculate how many of each should_see to sample for illusions
    # to match the should_see distribution in validation random
    samples_per_should_see = {}
    for key, count in val_should_see.items():
        if key.startswith('random →'):
            should_see_cls = key.split('→')[1].strip()
            samples_per_should_see[should_see_cls] = count
    
    print(f"\n{'='*70}")
    print(f"ILLUSION SAMPLING STRATEGY")
    print(f"{'='*70}")
    print(f"Total random in validation: {random_count_in_val}")
    print(f"Samples needed per illusion class: {random_count_in_val}")
    print(f"\nShould_see distribution to match:")
    for see_cls, cnt in samples_per_should_see.items():
        print(f"  {see_cls}: {cnt} samples")
    
    # Sample all_in and all_out with matching should_see distribution
    rng = np.random.default_rng(seed)
    test_illusion_indices = []
    
    for illusion_cls in ["all_in", "all_out"]:
        # Group indices by should_see
        indices_by_should_see = {}
        for idx in illusion_indices[illusion_cls]:
            _, _, cls_name, should_see = full_dataset[idx]
            if should_see not in indices_by_should_see:
                indices_by_should_see[should_see] = []
            indices_by_should_see[should_see].append(idx)
        
        # Sample from each should_see group
        for should_see_cls, count_needed in samples_per_should_see.items():
            available = indices_by_should_see.get(should_see_cls, [])
            if len(available) < count_needed:
                print(f"WARNING: {illusion_cls} has only {len(available)} samples "
                      f"for should_see={should_see_cls}, need {count_needed}")
                sampled = available
            else:
                sampled = rng.choice(available, count_needed, replace=False)
            test_illusion_indices.extend(sampled)
    
    # ================================================================
    # STEP 5: Create test set = validation + sampled illusions
    # ================================================================
    test_indices = np.concatenate([val_indices, test_illusion_indices])
    
    test_class_counts, test_should_see = analyze_distribution(
        test_indices, full_dataset, "TEST"
    )
    
    # ================================================================
    # STEP 6: Create DataLoaders
    # ================================================================
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    test_set = Subset(full_dataset, test_indices)
    
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Training batches:   {len(trainloader)}")
    print(f"Validation batches: {len(valloader)}")
    print(f"Test batches:       {len(testloader)}")
    
    return trainloader, valloader, testloader, full_dataset.class_to_idx


def test_dataloader_batch(loader, loader_name, class_to_idx):
    """Test that a dataloader produces valid batches"""
    print(f"\n{'='*70}")
    print(f"TESTING {loader_name} DATALOADER")
    print(f"{'='*70}")
    
    batch_count = 0
    total_samples = 0
    class_counts = Counter()
    
    for images, labels, cls_names, should_see in loader:
        batch_count += 1
        total_samples += len(images)
        
        for cls_name in cls_names:
            class_counts[cls_name] += 1
        
        if batch_count == 1:
            print(f"First batch:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Sample classes: {cls_names[:5]}")
            print(f"  Sample should_see: {should_see[:5]}")
    
    print(f"\nTotal: {batch_count} batches, {total_samples} samples")
    print(f"Class distribution across all batches:")
    for cls_name in sorted(class_counts.keys()):
        print(f"  {cls_name:15s}: {class_counts[cls_name]:5d} samples")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data/visual_illusion_dataset"
    BATCH_SIZE = 40
    SEED = 42
    
    print("\n" + "="*70)
    print("ILLUSION DATASET LOADER TEST")
    print("="*70)
    
    # Create dataloaders
    trainloader, valloader, testloader, class_to_idx = create_balanced_dataloaders(
        DATA_DIR, 
        batch_size=BATCH_SIZE, 
        seed=SEED
    )
    
    # Test each loader
    test_dataloader_batch(trainloader, "TRAIN", class_to_idx)
    test_dataloader_batch(valloader, "VALIDATION", class_to_idx)
    test_dataloader_batch(testloader, "TEST", class_to_idx)
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
