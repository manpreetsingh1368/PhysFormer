# physformer/dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
from .tokenizer import SimpleTokenizer

class LargeScaleImageDataset(Dataset):
    def __init__(self, folder_path, img_size=(720,1280), max_len=16):
        self.files = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if f.lower().endswith((".png",".jpg",".jpeg"))]
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
        self.tokenizer = SimpleTokenizer()
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        prompt_text = os.path.splitext(os.path.basename(img_path))[0]
        tokens = torch.tensor(self.tokenizer.encode(prompt_text, self.max_len), dtype=torch.long)
        return tokens, img
