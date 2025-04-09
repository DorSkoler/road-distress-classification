# Road Distress Classification

A deep learning project for automated classification of road surface distresses using computer vision techniques.

## Project Overview

This project implements a road distress classification system that can identify and classify various types of road surface damage. The system uses deep learning techniques, specifically leveraging transfer learning with EfficientNet-B3 and ResNet-50 architectures, to achieve accurate classification results.

## Features

- Multi-class classification of road distresses
- Support for multiple pre-trained backbone architectures (EfficientNet-B3, ResNet-50)
- Comprehensive data augmentation pipeline
- Training with learning rate scheduling and early stopping
- Detailed visualization of training metrics and results
- Dataset analysis and preprocessing tools

## Project Structure

```
road-distress-classification/
├── src/                      # Source code
│   ├── model.py             # Model architecture definitions
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── train.py             # Training script
│   └── train_multiple.py    # Multi-model training script
├── experiments/             # Experiment results and logs
├── visualizations/          # Generated plots and visualizations
├── notebooks/              # Jupyter notebooks for analysis
├── organized_dataset/      # Processed and organized dataset
├── checkpoints/           # Model checkpoints
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DorSkoler/road-distress-classification.git
cd road-distress-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset consists of road surface images labeled with various types of distresses. The data is organized with the following attributes:
- Damage classification
- Occlusion levels
- Crop variations

Use the `dataset_analysis.py` script to analyze the dataset distribution and characteristics.

## Training

To train a single model:
```bash
python src/train.py
```

To train multiple models with different configurations:
```bash
python src/train_multiple.py
```

## Results Visualization

The project includes comprehensive visualization tools:
- Training/validation curves
- Confusion matrices
- ROC curves
- Precision-Recall curves

Run the visualization script:
```bash
python visualize_results.py
```

## Model Performance

The system achieves competitive performance on road distress classification tasks. Detailed metrics and comparisons between different architectures can be found in the `experiments/research_summary.txt` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or feedback, please open an issue in the GitHub repository. 