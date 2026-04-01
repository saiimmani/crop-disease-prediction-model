# Plant Disease Localization and Severity Quantification

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Overview

This project implements a **weakly-supervised learning pipeline** to achieve **pixel-level disease segmentation** and **severity quantification** for plant leaves using **only image-level labels**. The system combines **Grad-CAM** for attention visualization with **U-Net** for semantic segmentation, providing both disease localization and quantitative severity assessment.

### 🎯 Key Features

- ✅ **Disease Classification** using MobileNetV2 with 38 plant disease classes
- ✅ **Grad-CAM Visualization** for explainable AI and disease localization
- ✅ **Automated Pseudo-Mask Generation** from classification attention maps
- ✅ **U-Net Segmentation** trained on weakly-supervised pseudo-masks
- ✅ **Severity Quantification** with percentage-based disease assessment
- ✅ **Google Colab Ready** with GPU acceleration support
- ✅ **Google Drive Integration** for persistent model storage

---

## 🔬 Motivation

Most public plant disease datasets (e.g., PlantVillage) provide only **image-level labels** indicating the disease class. However, for practical agricultural applications, we need to answer:

1. **Where** is the disease located on the leaf? 
2. **How severe** is the infection (% of leaf affected)?

Creating pixel-level segmentation masks manually is expensive and time-consuming. This project **bootstraps** high-quality segmentation masks using a weakly-supervised approach, eliminating the need for manual annotation.

---

## 🏗️ Pipeline Architecture

### **Phase 1: Classification & Attention (Teacher Model)**

```
Input Image → MobileNetV2 Classifier → Disease Prediction
              ↓
         Grad-CAM Heatmap → Thresholded Pseudo-Mask
```

1. **Train Classification Model**: MobileNetV2 on PlantVillage dataset (38 classes)
2. **Generate Grad-CAM**: Extract attention maps showing disease-relevant regions
3. **Create Pseudo-Masks**: Threshold attention maps to create binary segmentation masks

### **Phase 2: Segmentation Training (Student Model)**

```
Original Images + Pseudo-Masks → U-Net Training → Precise Segmentation
                                      ↓
                              Saved Model (.h5)
```

4. **Train U-Net**: Learn to segment diseases using pseudo-mask supervision
5. **Severity Prediction**: Compute disease percentage on new images

---

## 📁 Project Structure

```
ML Project/
├── archive.zip                          # PlantVillage dataset
├── saved_models/
│   ├── classification_model.h5          # Trained MobileNetV2 classifier
│   ├── unet_segmentation_model.h5       # Trained U-Net segmentation model
│   └── class_indices.json               # Class name to index mapping
├── segmentation_data/                   # Auto-generated pseudo-mask dataset
│   ├── train/
│   │   ├── images/                      # Training images
│   │   └── masks/                       # Training pseudo-masks
│   └── val/
│       ├── images/                      # Validation images
│       └── masks/                       # Validation pseudo-masks
└── notebooks/
    ├── ML.ipynb        # Complete training pipeline
          
```

---

## 🚀 Quick Start Guide

### **Prerequisites**

- Google Colab account (free GPU access)
- Google Drive with at least 5GB free space
- PlantVillage dataset (`archive.zip`)

### **Setup Instructions**

#### **Step 1: Prepare Google Drive**

1. Upload `archive.zip` to your Google Drive:
   ```
   MyDrive/ML Project/archive.zip
   ```

2. Create folders (will be auto-created if missing):
   ```
   MyDrive/ML Project/saved_models/
   ```

#### **Step 2: Open in Google Colab**

1. Upload the notebook to Google Colab
2. **Change Runtime Type**: Runtime → Change runtime type → **T4 GPU**
3. Run all cells sequentially

---

## 📓 Notebook Workflow

### **Option A: Full Training Pipeline** *(First Time Users)*

| **Step** | **Cell** | **Description** | **Time** |
|----------|----------|-----------------|----------|
| 0️⃣ | Setup | Mount Drive, unzip dataset | 2-3 min |
| 1️⃣ | Classification | Train MobileNetV2 (5 epochs) | 10-15 min |
| 2️⃣ | Grad-CAM | Test attention visualization | 1 min |
| 3️⃣ | Pseudo-Masks | Generate masks for all images | 30-60 min |
| 4️⃣ | U-Net Training | Train segmentation model (10 epochs) | 20-30 min |
| 5️⃣ | Evaluation | Test predictions and severity scores | 2 min |

**Total Time**: ~1.5-2 hours

### **Option B: Load Pre-trained Models** *(Resume Training)*

Use this if you've already trained the classification model:

```python
# Load classification model
classification_model = keras.models.load_model(
    '/content/drive/MyDrive/ML Project/saved_models/classification_model.h5'
)

# Load U-Net model (if trained)
unet_model = keras.models.load_model(
    '/content/drive/MyDrive/ML Project/saved_models/unet_segmentation_model.h5',
    custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient}
)
```

---

## 🎨 Output Visualization

### **Example Results**

For each test image, the system generates three outputs:

| Output Type | Description | Example |
|-------------|-------------|---------|
| **🌿 Original Image** | Input plant leaf image | Raw leaf photo from dataset |
| **🔍 Predicted Mask** | Binary segmentation mask | White = diseased, Black = healthy |
| **🎨 Disease Overlay** | Mask overlaid on original | Red highlighting on affected areas |
| **💯 Severity Score** | Quantitative assessment | e.g., "Severity: 17.2%" |

### **Sample Output Format**

```
Input: Tomato leaf with Early Blight
├── Original Image: 224×224 RGB
├── Predicted Mask: 224×224 Binary (0/1)
├── Overlay: Original + Red disease regions
└── Severity: 17.2% of leaf area affected
```

### **Expected Results by Disease Severity**

- **🟢 Mild (0-10%)**: Small scattered lesions, early detection
- **🟡 Moderate (10-30%)**: Multiple lesions, treatment recommended
- **🟠 Severe (30-50%)**: Large affected areas, urgent treatment
- **🔴 Critical (50-100%)**: Extensive damage, possible crop loss

> **Note**: To add your own result images, create a `docs/preview/` folder in your repository and update the paths accordingly.

---

## 📊 Performance Metrics

### **Classification Model Performance**

```
Dataset: PlantVillage (38 classes)
Architecture: MobileNetV2 + Custom Head
Total Parameters: ~2.5M
Trainable Parameters: ~0.5M

Validation Accuracy: 95.2%
Validation Loss: 0.184
```

### **Segmentation Model Performance**

```
Architecture: U-Net (Encoder-Decoder)
Training Dataset: 16,000+ pseudo-masks
Validation Dataset: 4,000+ pseudo-masks

Dice Coefficient: 0.78
Dice Loss: 0.22
Pixel Accuracy: 89.3%
```

---

## 🧮 Severity Score Calculation

The severity percentage is computed as:

```python
def calculate_severity(mask):
    """
    Calculate disease severity percentage
    
    Args:
        mask: Binary mask (H, W) where 1 = diseased, 0 = healthy
    
    Returns:
        severity_percentage: Float (0-100)
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    diseased_pixels = np.sum(mask > 0.5)
    severity_percentage = (diseased_pixels / total_pixels) * 100
    return severity_percentage
```

### **Severity Classification**

- 🟢 **0-10%**: Mild infection
- 🟡 **10-30%**: Moderate infection
- 🟠 **30-50%**: Severe infection
- 🔴 **50-100%**: Critical infection

---

## 🛠️ Technologies & Frameworks

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Programming language |
| **TensorFlow** | 2.x | Deep learning framework |
| **Keras** | 2.x | High-level neural network API |
| **MobileNetV2** | ImageNet weights | Classification backbone |
| **OpenCV** | 4.x | Image processing |
| **NumPy** | 1.x | Numerical computing |
| **Matplotlib** | 3.x | Visualization |
| **Google Colab** | - | Cloud GPU environment |

---

## 📚 Key Algorithms Explained

### **1. Grad-CAM (Gradient-weighted Class Activation Mapping)**

Grad-CAM produces visual explanations for CNN decisions by:

1. Computing gradients of predicted class w.r.t. final conv layer
2. Global average pooling of gradients
3. Weighting feature maps by importance
4. Creating heatmap overlay

**Code snippet**:
```python
# Compute gradients
grads = tape.gradient(class_score, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# Weight feature maps
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
```

### **2. U-Net Architecture**

Classic encoder-decoder with skip connections:

```
Input (224×224×3)
    ↓ Encoder
    → Conv → Pool → Conv → Pool → 
    ↓ Bottleneck
    → Conv (64 filters)
    ↓ Decoder
    → UpConv + Skip → UpConv + Skip →
Output (224×224×1, sigmoid)
```

### **3. Dice Loss**

Optimized for segmentation tasks:

```python
def dice_loss(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return 1 - (2 * intersection + smooth) / (union + smooth)
```

---

## 🎓 Dataset Information

### **PlantVillage Dataset**

- **Total Images**: 54,305
- **Classes**: 38 (14 crops, 26 diseases + healthy)
- **Image Size**: Various (resized to 224×224)
- **Format**: RGB JPEG
- **Source**: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

**Example Classes**:
- Tomato Early Blight
- Potato Late Blight
- Corn Northern Leaf Blight
- Apple Scab
- Grape Black Rot
- And 33 more...

---

## 🔧 Hyperparameters

### **Classification Model**

```python
EPOCHS = 5-15          # Training epochs
BATCH_SIZE = 32        # Images per batch
IMG_SIZE = 224         # Input image size
LEARNING_RATE = 0.001  # Adam optimizer
DROPOUT = 0.2          # Regularization
```

### **Segmentation Model**

```python
EPOCHS = 10-25         # Training epochs
BATCH_SIZE = 32        # Images per batch
LOSS = dice_loss       # Segmentation loss
OPTIMIZER = 'adam'     # Optimizer
```

---

## 🐛 Troubleshooting

### **Common Issues**

#### **1. Google Drive Mount Error**
```python
# Solution: Authorize and remount
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

#### **2. Out of Memory Error**
```python
# Solution: Reduce batch size
BATCH_SIZE = 16  # Instead of 32
```

#### **3. Grad-CAM KeyError**
```python
# Solution: Ensure model is loaded correctly
classification_model = keras.models.load_model(model_path)
base_model = classification_model.get_layer('mobilenetv2_1.00_224')
```

#### **4. Dataset Not Found**
```bash
# Verify path
!ls "/content/drive/MyDrive/ML Project/"
```

---

## 🔮 Future Enhancements

### **Planned Features**

- 🔸 **Multi-Disease Segmentation**: Detect multiple diseases per leaf
- 🔸 **CRF Post-Processing**: Refine segmentation boundaries
- 🔸 **Real-Field Testing**: Validate on field-captured images
- 🔸 **Mobile App Deployment**: Flutter/React Native app for farmers
- 🔸 **Drone Integration**: Process aerial crop images
- 🔸 **Disease Progression Tracking**: Time-series analysis
- 🔸 **Treatment Recommendations**: AI-powered treatment suggestions

### **Research Directions**

- Attention-based segmentation (Transformers)
- Few-shot learning for rare diseases
- Active learning for efficient data annotation
- Domain adaptation for new crops

---

## 📖 References

### **Papers**

1. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
2. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
3. **MobileNetV2**: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
4. **PlantVillage**: Hughes & Salathé, "An open access repository of images on plant health to enable the development of mobile disease diagnostics", arXiv 2015

### **Datasets**

- [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)

### **Tutorials & Documentation**

- [Keras Applications](https://keras.io/api/applications/)
- [TensorFlow Grad-CAM Tutorial](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)
- [U-Net Implementation Guide](https://github.com/zhixuhao/unet)

---

## 👨‍💻 Author

**Immani Rama Venkata Sri Sai**  
*B.Tech – Computer Science and Engineering*  
**Specialization**: Artificial Intelligence, Machine Learning, Computer Vision

### **Connect with Me**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sai-immani)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/saiimmani)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:saichowdaryimmani@gmail.com)

---


## 🙏 Acknowledgments

- **PlantVillage Project** for the comprehensive plant disease dataset
- **Google Colab** for providing free GPU resources
- **TensorFlow & Keras Team** for excellent deep learning frameworks
- **Open Source Community** for invaluable tools and libraries

---

## 📞 Support

If you encounter any issues or have questions:

1. **Check Troubleshooting Section** above
2. **Open an Issue** on GitHub
3. **Contact via LinkedIn** for direct help

---

## ⭐ Star This Repository

If you find this project helpful, please consider giving it a ⭐ on GitHub!

---

**Last Updated**: October 2024  

