# scripts/generate_demo.py
import torch
from physformer.model import PhysFormer
from physformer.tokenizer import SimpleTokenizer
from physformer.utils import save_patch_image

device = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 32
IMG_W, IMG_H = 1280, 720
NUM_PATCHES_W = IMG_W // PATCH_SIZE
NUM_PATCHES_H = IMG_H // PATCH_SIZE
TOTAL_PATCHES = NUM_PATCHES_W * NUM_PATCHES_H

tokenizer = SimpleTokenizer()
model = PhysFormer(vocab_size=len(tokenizer.vocab), patch_dim=PATCH_SIZE*PATCH_SIZE).to(device)
model.load_state_dict(torch.load("checkpoints/physformer_checkpoint.pt", map_location=device))
model.eval()

prompt = input("Enter prompt: ")
tokens = torch.tensor([tokenizer.encode(prompt, max_len=16)], dtype=torch.long).to(device)
with torch.no_grad():
    with torch.cuda.amp.autocast():
        image = model(tokens, TOTAL_PATCHES)
save_patch_image(image, "physformer_output.png")
