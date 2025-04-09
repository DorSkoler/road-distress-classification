import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as F
import random
import json
import glob

class RoadDistressDataset(Dataset):
    """
    Dataset class for road distress classification
    """
    def __init__(self, data_dir, split='train', transform_type='train'):
        """
        Initialize the dataset
        
        Args:
            data_dir (str): Path to the dataset directory
            split (str): One of 'train', 'val', or 'test'
            transform_type (str): One of 'train' or 'val'
        """
        self.data_dir = data_dir
        self.split = split
        self.transform_type = transform_type
        
        # Load metadata
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
            
        # Get image and annotation directories
        self.images_dir = os.path.join(data_dir, 'images', split)
        self.annotations_dir = os.path.join(data_dir, 'annotations', split)
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(self.images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(self.image_files)} images in {split} set")
        
        # Define transforms
        self.transforms = self._get_transforms()
        
    def _get_transforms(self):
        """Get image transforms based on transform type"""
        if self.transform_type == 'train':
            return T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),  # Convert to tensor first
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                T.RandomPerspective(distortion_scale=0.2, p=0.5),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                T.RandomErasing(p=0.3, scale=(0.02, 0.2)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image path and annotation path
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        ann_path = os.path.join(self.annotations_dir, os.path.splitext(img_name)[0] + '.json')
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        
        # Load annotation
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
        
        # Get labels
        damage_tag = next((tag for tag in annotation['tags'] if tag['name'] == 'Damage'), None)
        occlusion_tag = next((tag for tag in annotation['tags'] if tag['name'] == 'Occlusion'), None)
        crop_tag = next((tag for tag in annotation['tags'] if tag['name'] == 'Crop'), None)
        
        # Convert labels to binary
        damage_label = 1 if damage_tag and damage_tag['value'] == 'Damaged' else 0
        occlusion_label = 1 if occlusion_tag and occlusion_tag['value'] == 'Occluded' else 0
        crop_label = 1 if crop_tag and crop_tag['value'] == 'Cropped' else 0
        
        # Create label tensor
        label = torch.tensor([damage_label, occlusion_label, crop_label], dtype=torch.float32)
        
        return image, label 

def test_data_loader():
    """Test function to verify the data loading pipeline"""
    # Test dataset initialization
    try:
        dataset = RoadDistressDataset(
            data_dir='organized_dataset',
            split='train',
            transform_type='train'
        )
        print("Dataset initialization successful!")
        
        # Test getting a single sample
        image, label = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Label: {label}")
        
        # Test data loader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch_images, batch_labels = next(iter(dataloader))
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        
        print("Data loading pipeline test completed successfully!")
        
    except Exception as e:
        print(f"Error in data loading pipeline: {str(e)}")
        raise

if __name__ == '__main__':
    test_data_loader() 