# scripts/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from physformer.model import PhysFormer
from physformer.dataset import LargeScaleImageDataset
from physformer.tokenizer import SimpleTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 32
IMG_W, IMG_H = 1280, 720
NUM_PATCHES_W = IMG_W // PATCH_SIZE
NUM_PATCHES_H = IMG_H // PATCH_SIZE
TOTAL_PATCHES = NUM_PATCHES_W * NUM_PATCHES_H

dataset = LargeScaleImageDataset("./data/train")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = PhysFormer(vocab_size=len(SimpleTokenizer().vocab), patch_dim=PATCH_SIZE*PATCH_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler()

for epoch in range(1):
    for tokens, target in loader:
        tokens, target = tokens.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(tokens, TOTAL_PATCHES)
            if output.shape != target.shape:
                target = F.interpolate(target, size=output.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "checkpoints/physformer_checkpoint.pt")
