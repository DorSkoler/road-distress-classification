from src.dataset_organization import OrganizedDatasetLoader
from src.exploratory_analysis import analyze_image_shapes
from src.preprocessing import resize_images

def main():
    # Define paths
    dataset_path = "./organized_dataset"
    output_folder = "./outputs/augmented_images"

    # Initialize dataset loader
    loader = OrganizedDatasetLoader(dataset_path)

    # Load training data
    print("Loading training data...")
    train_images, train_annotations = loader.load_split('train')

    # Perform exploratory analysis
    print("\nPerforming exploratory analysis...")
    analyze_image_shapes(train_images)

    # Print dataset statistics
    print("\nDataset statistics:")
    for split in ['train', 'val', 'test']:
        stats = loader.get_split_stats(split)
        print(f"\n{split.upper()} set:")
        print(f"Total images: {stats['count']}")
        print("Damage distribution:", stats['damage_distribution'])

    # Example preprocessing (resize images)
    print("\nPreprocessing images...")
    resized_images = resize_images(train_images, output_folder, size=(512, 512))

if __name__ == "__main__":
    main() 