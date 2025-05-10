import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import cv2

class RoadDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=256, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        # match crop images only if there is an annotated mask (suffix '_annotated.png' or '_full_annotated.png')
        all_images = sorted([f for f in os.listdir(images_dir) if f.endswith('_crop.png')])
        self.samples = []  # list of (image_name, mask_name)
        for img_name in all_images:
            base = img_name[:-9]  # remove '_crop.png'
            # look for masks with either '_annotated' or '_full_annotated'
            candidates = [m for m in os.listdir(masks_dir)
                          if m.startswith(base) and m.endswith('annotated.png')]
            if candidates:
                mask_name = candidates[0]
                self.samples.append((img_name, mask_name))
            else:
                print(f"Skipping image without mask: {img_name}")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, mask_name = self.samples[idx]
        image_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        # Resize to fixed dimensions
        if self.img_size is not None:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask / 255.0).unsqueeze(0).float()
        return image, mask


def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # Dataset and DataLoader
    dataset = RoadDataset(args.image_dir, args.mask_dir, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Model
    model = smp.Unet(
        encoder_name='resnet34', encoder_weights='imagenet', classes=1, activation=None
    )
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss functions and optimizer
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            # Combined loss: Dice + BCE
            loss = dice_loss(preds, masks) + bce_loss(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model: {ckpt_path}")
    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train road segmentation model')
    parser.add_argument('--image-dir', type=str, default='preprocessing/output', help='Directory of cropped images')
    parser.add_argument('--mask-dir', type=str, default='preprocessing/annotations', help='Directory of annotated masks')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Where to save model weights')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--img-size', type=int, default=256, help='Resize images and masks to this size')
    args = parser.parse_args()
    train(args) 