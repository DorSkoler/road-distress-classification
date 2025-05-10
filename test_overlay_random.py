import os
import random
import torch
import numpy as np
from PIL import Image
import cv2
import segmentation_models_pytorch as smp

# Settings
DATA_DIR = 'data/coryell'
MODEL_PATH = 'checkpoints/best_model.pth'
N_SAMPLES = 50

# Collect all raw image paths
img_paths = []
for road in os.listdir(DATA_DIR):
    img_folder = os.path.join(DATA_DIR, road, 'img')
    if not os.path.isdir(img_folder):
        continue
    for fn in os.listdir(img_folder):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            img_paths.append(os.path.join(img_folder, fn))

if len(img_paths) < N_SAMPLES:
    print(f"Only found {len(img_paths)} images, using all.")
    N_SAMPLES = len(img_paths)

sampled_paths = random.sample(img_paths, N_SAMPLES)

# Model setup
model = smp.Unet(
    encoder_name='resnet34', encoder_weights='imagenet', classes=1, activation=None
)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for img_path in sampled_paths:
    orig = Image.open(img_path).convert('RGB')
    orig_np = np.array(orig)
    # Resize for model
    image = orig.resize((256, 256))
    img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = torch.sigmoid(pred).cpu().numpy()[0, 0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        # Upsample mask to original image size
        pred_mask_up = cv2.resize(pred_mask, (orig_np.shape[1], orig_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Overlay: green mask with alpha
    overlay = orig_np.copy()
    green = np.zeros_like(overlay)
    green[..., 1] = 255
    mask_bool = pred_mask_up > 127
    overlay[mask_bool] = cv2.addWeighted(overlay, 0.5, green, 0.5, 0)[mask_bool]

    # Show overlay
    cv2.imshow('Prediction Overlay', overlay)
    print(f"Showing: {img_path}")
    key = cv2.waitKey(0)
    if key == 27:  # ESC to quit early
        break

cv2.destroyAllWindows()