# Road Dataset Pipeline

This pipeline processes raw images to create a filtered dataset containing only images with sufficient road content (≥15% road area). It uses a pretrained segmentation model to identify road areas and filters images accordingly.

## Directory Structure

```
road_dataset_pipeline/
├── raw/                # Place your raw images here
├── masks/              # Temporary storage for predicted masks
├── filtered/           # Images passing the road threshold
│   ├── train/         # Training set (specific roads)
│   ├── val/           # Validation set (different roads)
│   └── test/          # Test set (unseen roads)
├── filtered_masks/     # Corresponding masks for filtered images
│   ├── train/
│   ├── val/
│   └── test/
├── scripts/           # Pipeline scripts
└── tagged_json/       # (Optional) For additional labels
```

## Usage

1. **Prepare Raw Images**
   - Place your raw images in the `raw/` directory
   - Images should be organized by roads (e.g., `road_1/`, `road_2/`, etc.)
   - Supported formats: PNG, JPG, JPEG

2. **Filter Images by Road Content**
   ```bash
   cd scripts
   python filter_by_road.py
   ```
   This will:
   - Process all images in `raw/`
   - Generate road masks using the pretrained model
   - Save images with ≥15% road area to `filtered/`
   - Save corresponding masks to `filtered_masks/`

3. **Split Dataset by Roads**
   ```bash
   cd scripts
   python split_dataset.py
   ```
   This will:
   - Split images based on their road folders
   - Train on specific roads (e.g., roads 1-3)
   - Validate on different roads (e.g., roads 4-5)
   - Test on completely unseen roads (e.g., road 6)
   - This ensures the model is tested on roads it hasn't seen before

## Configuration

You can modify the following parameters in the scripts:

- `filter_by_road.py`:
  - `ROAD_THRESHOLD`: Minimum road area percentage (default: 0.15)
  - `IMG_SIZE`: Input size for the model (default: 256)

- `split_dataset.py`:
  - `SPLITS`: Road assignments for each split (modify based on your road folders)
  ```python
  SPLITS = {
      'train': ['road_1', 'road_2', 'road_3'],  # Roads for training
      'val': ['road_4', 'road_5'],              # Roads for validation
      'test': ['road_6']                        # Roads for testing
  }
  ```

## Requirements

- Python 3.6+
- PyTorch
- segmentation-models-pytorch
- OpenCV
- Pillow
- tqdm

## Notes

- The pipeline uses a pretrained U-Net model with ResNet34 encoder
- All images are resized to 256x256 for processing
- Original image dimensions are preserved in the output
- Progress bars show processing status
- Error handling is included for robustness
- Dataset is split by roads to ensure proper evaluation on unseen roads 