# Sports Image Classification

This repository contains a modular PyTorch framework for training and running inference on sports image datasets.

## Project Structure

```
sports-image-classification/
├── config/
│   └── config.yaml          # Hyperparameters and folder paths configuration
├── dataset/                 # Dataset folder (should contain train, val, and test subfolders)
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Custom Dataset class and DataLoader setup
│   ├── model.py             # Model initialization (ResNet50, EfficientNet, MobileNet)
│   ├── train.py             # Training and validation loops
│   ├── inference.py         # Prediction script for single images
│   └── utils.py             # Helper tools (seeding, configuration loading, plotting)
├── requirements.txt         # Required Python packages
└── README.md                # Documentation
```

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset:**
   Place your images in `dataset/` under the respective `train`, `val`, and `test` directories matching your categories:
   ```
   dataset/
   ├── train/
   │   ├── football/
   │   └── basketball/
   ├── val/
   │   ├── football/
   │   └── basketball/
   └── test/
       ├── football/
       └── basketball/
   ```

3. **Train the Model:**
   ```bash
   python src/train.py
   ```

4. **Run Prediction:**
   ```bash
   python src/inference.py --image path/to/image.jpg
   ```
