import os
from pathlib import Path
from src.dataset_organization import organize_dataset, OrganizedDatasetLoader

def main():
    # Define paths
    source_path = r"C:\data\road-distress\coryell"
    output_path = "./organized_dataset"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Organize the dataset
    print("Organizing dataset...")
    metadata = organize_dataset(source_path, output_path)
    
    # Print dataset statistics
    print("\nDataset organization complete!")
    print(f"Total images: {metadata['total_images']}")
    for split, stats in metadata['splits'].items():
        print(f"\n{split.upper()} set:")
        print(f"Count: {stats['count']}")
        print("Damage distribution:", stats['damage_distribution'])
    
    # Test the loader
    print("\nTesting dataset loader...")
    print("Initializing loader...")
    loader = OrganizedDatasetLoader(output_path)
    print("Loader initialized successfully")
    
    # Load and print statistics for each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        images, annotations = loader.load_split(split)
        print(f"Loaded {len(images)} images from {split} set")
        
        # Print some sample annotations
        if annotations:
            print("Sample annotation:", annotations[0])
        print(f"Completed processing {split} split")

if __name__ == "__main__":
    main() 