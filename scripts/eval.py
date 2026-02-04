# scripts/eval.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from physformer.model import PhysFormer
from physformer.dataset import LargeScaleImageDataset
from physformer.tokenizer import SimpleTokenizer
from physformer.utils import save_patch_image

device = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 32
IMG_W, IMG_H = 1280, 720
NUM_PATCHES_W = IMG_W // PATCH_SIZE
NUM_PATCHES_H = IMG_H // PATCH_SIZE
TOTAL_PATCHES = NUM_PATCHES_W * NUM_PATCHES_H

# Load dataset
dataset = LargeScaleImageDataset("./data/test")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load model
tokenizer = SimpleTokenizer()
model = PhysFormer(vocab_size=len(tokenizer.vocab), patch_dim=PATCH_SIZE*PATCH_SIZE).to(device)
model.load_state_dict(torch.load("checkpoints/physformer_checkpoint.pt", map_location=device))
model.eval()

mse_total = 0.0
num_samples = len(loader)

for i, (tokens, target) in enumerate(loader):
    tokens, target = tokens.to(device), target.to(device)
    with torch.no_grad():
        output = model(tokens, TOTAL_PATCHES)
        if output.shape != target.shape:
            target = F.interpolate(target, size=output.shape[2:], mode='bilinear', align_corners=False)
        mse = F.mse_loss(output, target).item()
        mse_total += mse

    # Optional: save some generated images for inspection
    if i < 5:
        save_patch_image(output, f"eval_output_{i}.png")

avg_mse = mse_total / num_samples
print(f"Evaluation done. Average MSE: {avg_mse:.6f}")
