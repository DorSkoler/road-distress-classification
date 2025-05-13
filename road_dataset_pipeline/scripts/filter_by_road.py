import os
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import cv2
from tqdm import tqdm
import shutil

# Configuration
MODEL_PATH = '../../checkpoints/best_model.pth'
RAW_DIR = '../raw'
MASKS_DIR = '../masks'
FILTERED_IMG_DIR = '../filtered'
FILTERED_MASK_DIR = '../filtered_masks'
ROAD_THRESHOLD = 0.15
IMG_SIZE = 256

def ensure_dirs():
    """Ensure all required directories exist"""
    for dir_path in [MASKS_DIR, FILTERED_IMG_DIR, FILTERED_MASK_DIR]:
        os.makedirs(dir_path, exist_ok=True)

def load_model():
    """Load the pretrained segmentation model"""
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def process_image(model, img_path, device):
    """Process a single image and return road area percentage"""
    # Load and preprocess image
    image = Image.open(img_path).convert('RGB')
    orig_np = np.array(image)
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    # Get prediction
    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = torch.sigmoid(pred).cpu().numpy()[0, 0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        pred_mask_up = cv2.resize(pred_mask, (orig_np.shape[1], orig_np.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)

    # Calculate road percentage
    road_ratio = np.mean(pred_mask_up)
    return road_ratio, pred_mask_up, orig_np

def main():
    # Setup
    ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model().to(device)
    
    # Process all images
    image_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    filtered_count = 0
    
    print(f"Processing {len(image_files)} images...")
    for img_name in tqdm(image_files):
        img_path = os.path.join(RAW_DIR, img_name)
        try:
            road_ratio, pred_mask, orig_img = process_image(model, img_path, device)
            
            if road_ratio >= ROAD_THRESHOLD:
                # Save filtered image and mask
                base_name = os.path.splitext(img_name)[0]
                Image.fromarray(orig_img).save(os.path.join(FILTERED_IMG_DIR, img_name))
                Image.fromarray((pred_mask * 255).astype(np.uint8)).save(
                    os.path.join(FILTERED_MASK_DIR, f"{base_name}_mask.png"))
                filtered_count += 1
                
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
    
    print(f"\nProcessing complete!")
    print(f"Total images processed: {len(image_files)}")
    print(f"Images passing {ROAD_THRESHOLD*100}% road threshold: {filtered_count}")
    print(f"Filtered images saved to: {FILTERED_IMG_DIR}")
    print(f"Corresponding masks saved to: {FILTERED_MASK_DIR}")

if __name__ == '__main__':
    main() 