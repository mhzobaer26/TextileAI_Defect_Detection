"""
Quick Model Training Script
This will train a compatible model for the web app
"""

print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║        🧵 TextileAI - Quick Model Training                       ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from datetime import datetime

print("\n📦 Loading TensorFlow...")
print(f"   Version: {tf.__version__}")

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # Reduced for quick training
LEARNING_RATE = 0.001

print("\n⚙️  Configuration:")
print(f"   Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS} (quick training)")
print(f"   Learning Rate: {LEARNING_RATE}")

# Check if dataset exists
DATASET_PATH = 'Dataset'
if not os.path.exists(DATASET_PATH):
    print(f"\n❌ Dataset not found: {DATASET_PATH}")
    print("\n💡 Creating synthetic mini dataset for testing...")
    
    # Create minimal dataset structure
    for split in ['train', 'validation']:
        for class_name in ['defect', 'no_defect']:
            path = os.path.join(DATASET_PATH, split, class_name)
            os.makedirs(path, exist_ok=True)
    
    # Generate 20 synthetic images for testing
    import cv2
    
    print("   Generating synthetic images...")
    for split in ['train', 'validation']:
        num_images = 10 if split == 'train' else 5
        for class_name in ['defect', 'no_defect']:
            path = os.path.join(DATASET_PATH, split, class_name)
            for i in range(num_images):
                # Create synthetic fabric texture
                img = np.random.randint(200, 250, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                
                # Add pattern
                for j in range(0, IMG_SIZE, 4):
                    cv2.line(img, (j, 0), (j, IMG_SIZE), (180, 180, 180), 1)
                    cv2.line(img, (0, j), (IMG_SIZE, j), (180, 180, 180), 1)
                
                # Add defect for defect class
                if class_name == 'defect':
                    cv2.circle(img, (IMG_SIZE//2, IMG_SIZE//2), 30, (100, 100, 100), -1)
                
                cv2.imwrite(os.path.join(path, f'sample_{i}.jpg'), img)
    
    print("   ✅ Synthetic dataset created!")

# Create data generators
print("\n📊 Loading dataset...")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'validation'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print(f"   Train samples: {train_generator.samples}")
print(f"   Validation samples: {validation_generator.samples}")

# Build model
print("\n🏗️  Building model...")

def create_model():
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')
    ])
    
    return model

model = create_model()

print("   ✅ Model created!")
print(f"   Total parameters: {model.count_params():,}")

# Compile model
print("\n⚙️  Compiling model...")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("   ✅ Model compiled!")

# Train model
print(f"\n🚀 Training model for {EPOCHS} epochs...")
print("   (This may take a few minutes...)\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=1
)

# Save model
MODEL_PATH = 'textile_defect_model.keras'
print(f"\n💾 Saving model...")

model.save(MODEL_PATH)

print(f"   ✅ Model saved: {MODEL_PATH}")
print(f"   Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")

# Verify model can be loaded
print(f"\n🔍 Verifying model...")

try:
    test_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"   ✅ Model verified - compatible with TensorFlow {tf.__version__}")
except Exception as e:
    print(f"   ❌ Error: {str(e)}")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

print(f"\n📊 Final Results:")
print(f"   Training Accuracy: {history.history['accuracy'][-1]:.2%}")
print(f"   Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")

print(f"\n🚀 Next Steps:")
print(f"   1. Model is ready: {MODEL_PATH}")
print(f"   2. Run the web app: streamlit run app.py")
print(f"   3. Upload fabric images for defect detection!")

print("\n💡 Note: This was a quick training run.")
print("   For production, train longer (50+ epochs) with your actual dataset.")

print("\n" + "="*70 + "\n")
