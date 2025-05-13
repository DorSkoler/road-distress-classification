import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from tqdm import tqdm

class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get all image-mask pairs
        self.samples = []
        for split in ['train', 'val', 'test']:
            img_split_dir = os.path.join(image_dir, split)
            mask_split_dir = os.path.join(mask_dir, split)
            
            if not os.path.exists(img_split_dir):
                continue
                
            for img_name in os.listdir(img_split_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    base_name = os.path.splitext(img_name)[0]
                    mask_name = f"{base_name}_mask.png"
                    
                    if os.path.exists(os.path.join(mask_split_dir, mask_name)):
                        self.samples.append({
                            'image': os.path.join(img_split_dir, img_name),
                            'mask': os.path.join(mask_split_dir, mask_name),
                            'split': split
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image and mask
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        
        # Resize to model input size
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return image, mask

def train_model():
    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and dataloader
    dataset = RoadDataset(
        image_dir='../filtered',
        mask_dir='../filtered_masks'
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Create model
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        classes=1,
        activation=None
    ).to(DEVICE)
    
    # Loss functions
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"Training on {DEVICE}")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            # Forward pass
            preds = model(images)
            
            # Calculate loss
            loss = dice_loss(preds, masks) + bce_loss(preds, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'../checkpoints/model_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), '../checkpoints/final_model.pth')
    print("Training complete!")

if __name__ == '__main__':
    train_model() 