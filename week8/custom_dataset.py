from PIL import Image
from torch.utils.data import Dataset

class MyDataset1(Dataset):
    def __init__(self, txt_path, transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1]), float(words[2])))
        self.imgs = imgs        
        self.transform = transform

    def __getitem__(self, index):
        fn, label,noise = self.imgs[index]
        img = Image.open(fn).convert('RGB')    
        if self.transform is not None:
            img = self.transform(img)  
        return img, label,noise

    def __len__(self):
        return len(self.imgs)
