import os
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

# Paths
MODEL_PATH = 'checkpoints/best_model.pth'
IMAGE_DIR = 'preprocessing/output'  # directory with *_crop.png images
OUTPUT_DIR = 'test_predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model setup (must match training)
model = smp.Unet(
    encoder_name='resnet34', encoder_weights='imagenet', classes=1, activation=None
)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Inference loop
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.endswith('_crop.png'):
        continue
    img_path = os.path.join(IMAGE_DIR, img_name)
    image = Image.open(img_path).convert('RGB')
    image = image.resize((256, 256))  # Must match training size
    img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = torch.sigmoid(pred).cpu().numpy()[0, 0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # Save predicted mask
    out_name = img_name.replace('_crop.png', '_pred.png')
    out_path = os.path.join(OUTPUT_DIR, out_name)
    Image.fromarray(pred_mask).save(out_path)
    print(f"Saved: {out_path}") 