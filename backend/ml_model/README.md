# Child Face Emotion Recognition Model

Lightweight emotion detection model optimized for hearing therapy application.

## 📁 Directory Structure

```
ml_model/
├── dataset/                    # Your training images
│   ├── happy/
│   ├── sad/
│   ├── confused/
│   ├── neutral/
│   ├── excited/
│   └── frustrated/
├── saved_models/              # Trained models
│   └── emotion_model_TIMESTAMP/
│       ├── best_model.h5
│       ├── emotion_model.tflite
│       ├── metadata.json
│       └── training_history.png
├── train_emotion_model.py     # Training script
├── emotion_detector.py        # Inference script
└── requirements_ml.txt        # ML dependencies
```

## 🚀 Quick Start

### Step 1: Organize Your Dataset

Place your child face emotion images in the dataset folders:

```
dataset/
├── happy/        # Images of happy children
├── sad/          # Images of sad children
├── confused/     # Images of confused children
├── neutral/      # Images of neutral expressions
├── excited/      # Images of excited children
└── frustrated/   # Images of frustrated children
```

**Recommendations:**
- Minimum 100 images per emotion (more is better!)
- Images should be clear and show the face prominently
- Mix of lighting conditions, angles, and backgrounds
- Diverse set of children (different ages, skin tones, etc.)

### Step 2: Install Dependencies

```bash
cd backend/ml_model
pip install -r requirements_ml.txt
```

### Step 3: Train the Model

```bash
python train_emotion_model.py
```

**Training Options:**
- Edit `use_transfer_learning=False` to `True` in the script for MobileNetV2 (better accuracy)
- Adjust `EPOCHS`, `BATCH_SIZE`, `IMG_SIZE` at the top of the script
- Model automatically saves best version based on validation accuracy

**Training Output:**
- `saved_models/emotion_model_TIMESTAMP/` contains all model files
- `training_history.png` shows accuracy/loss curves
- `confusion_matrix.png` shows per-class performance
- `metadata.json` contains model configuration

### Step 4: Test the Model

```bash
python emotion_detector.py
```

This will test the trained model on sample images from your dataset.

## 📊 Model Architecture

### Custom Lightweight CNN (Default)
- Optimized for real-time inference
- ~500K parameters (~2MB model size)
- Uses depthwise separable convolutions
- Input: 96x96 RGB images
- Output: 6 emotion classes

### MobileNetV2 Transfer Learning (Alternative)
- Higher accuracy, slightly larger size
- ~1M parameters (~4MB model size)
- Pre-trained on ImageNet
- Fine-tuned for emotion recognition

## 🎯 Emotion Classes

1. **happy** - Child is happy and engaged
2. **sad** - Child appears sad or disappointed
3. **confused** - Child is puzzled or uncertain
4. **neutral** - Neutral expression, focused
5. **excited** - Child is very enthusiastic
6. **frustrated** - Child is struggling or annoyed

## 🔧 Integration with FastAPI

The emotion detector will be integrated into your backend to provide:
- Real-time emotion detection during gameplay
- Personalized feedback based on child's emotional state
- Adaptive difficulty based on frustration levels

## 📈 Performance Metrics

After training, check these files in your model directory:

- **training_history.png** - Monitor overfitting (train vs val curves)
- **confusion_matrix.png** - See which emotions are confused
- **metadata.json** - Check test accuracy

**Target Performance:**
- Validation accuracy: >80% (good)
- Validation accuracy: >90% (excellent)

## 🎨 Data Augmentation

The training script automatically applies:
- Random rotation (±20°)
- Width/height shifts (±20%)
- Horizontal flips
- Zoom (±20%)
- Brightness variations

This helps the model generalize better to:
- Different lighting conditions
- Various camera angles
- Head tilts and positions

## 💡 Tips for Better Results

### 1. Data Quality
- Ensure images are properly labeled
- Remove blurry or unclear images
- Balance classes (similar number of images per emotion)

### 2. Data Quantity
- Start with at least 100 images per emotion
- Ideally 300-500 per emotion for best results
- Consider data augmentation for small datasets

### 3. Training Tips
- If validation loss stops improving, training will stop early
- If overfitting (train accuracy >> val accuracy), add more data
- Monitor confusion matrix to see problematic classes

### 4. Real-time Performance
- Use `.tflite` model for fastest inference
- 96x96 input size balances accuracy and speed
- Model runs at 30+ FPS on modern hardware

## 🔄 Model Updates

To retrain with new data:

1. Add new images to dataset folders
2. Run training script again
3. New model saved with timestamp
4. Update FastAPI to use latest model

## 🐛 Troubleshooting

**"No images found in dataset"**
- Check that images are in correct folders
- Supported formats: .jpg, .jpeg, .png

**Low validation accuracy (<60%)**
- Need more training data
- Check if images are properly labeled
- Try transfer learning (MobileNetV2)

**Model too slow**
- Use TFLite model instead of Keras
- Reduce input size (edit IMG_SIZE)
- Consider model quantization

**High accuracy on train, low on validation**
- Overfitting: need more diverse data
- Try adding more augmentation
- Reduce model complexity

## 📝 License & Usage

This model is specifically designed for educational hearing therapy applications. Ensure compliance with privacy regulations when collecting and using child face images.

## 🤝 Support

For issues or questions, refer to the main project documentation or create an issue in the repository.
