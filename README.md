# Spatially Adaptive Uncertainty Modeling for Robust Medical Image Segmentation

## Requirements

The **Python packages** needed are listed below.

```
torch==1.10.1
torchvision==0.11.2
timm==0.6.12
monai==0.8.1
thop==0.1.1.post2209072238
numpy==1.18.5
pandas==1.1.5
scipy==1.5.4
scikit-image==0.17.2
scikit-learn==0.23.2
albumentations==1.0.3
opencv-python==4.4.0.46
SimpleITK==2.1.1.2
MedPy==0.4.0
Pillow==8.0.1
```

### Data preparation

Resize datasets( [TN3k](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation),[TG3k](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation) and [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)) to 224*224 and saved them in npy format.

```
python data_preprocess.py
```

```
├── TN3k
    ├──image
    	├──001.npy
    ├──label
    	├──001.npy
```

### Train and Test

Our method is easy to train and test, just need to run "train_and_test_isic.py".

```
python train_and_test_isic.py
```
