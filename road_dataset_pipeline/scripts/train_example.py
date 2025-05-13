import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt

def visualize_training_step():
    """Visualize how model learns from road masks"""
    # 1. Create a simple image and mask
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    # Add a road-like shape
    mask[100:150, 50:200] = 255  # White road area
    
    # 2. Convert to tensors
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
    
    # 3. Create a simple model
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        classes=1,
        activation=None
    )
    
    # 4. Forward pass
    with torch.no_grad():
        pred = model(image_tensor.unsqueeze(0))
        pred = torch.sigmoid(pred)  # Convert to probabilities
    
    # 5. Visualize
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth\n(White = Road)')
    plt.axis('off')
    
    # Model prediction
    plt.subplot(133)
    plt.imshow(pred[0, 0].numpy(), cmap='gray')
    plt.title('Model Prediction\n(Brighter = Higher Road Probability)')
    plt.axis('off')
    
    plt.suptitle('How Model Learns Road Areas', y=1.05)
    plt.tight_layout()
    plt.show()
    
    # 6. Print explanation
    print("\nTraining Process:")
    print("1. Input: Image of road")
    print("2. Ground Truth: White pixels = road areas")
    print("3. Model Output: Probability of each pixel being road")
    print("4. Loss: Compare prediction with ground truth")
    print("   - High probability for white pixels = good")
    print("   - Low probability for black pixels = good")
    print("5. Update: Adjust model to minimize loss")

if __name__ == '__main__':
    visualize_training_step() 