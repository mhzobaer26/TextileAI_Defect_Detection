"""
Create a Simple Compatible Model for TensorFlow 2.20.0
Using custom CNN instead of EfficientNet to avoid BatchNorm issues
"""

print("\n🔧 Creating TensorFlow 2.20.0 Compatible Model...\n")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from datetime import datetime

print(f"TensorFlow Version: {tf.__version__}")

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

# Check/Create dataset
DATASET_PATH = 'Dataset'
if not os.path.exists(DATASET_PATH):
    print("\n📊 Creating synthetic dataset...")
    import cv2
    
    for split in ['train', 'validation']:
        for class_name in ['defect', 'no_defect']:
            path = os.path.join(DATASET_PATH, split, class_name)
            os.makedirs(path, exist_ok=True)
    
    for split in ['train', 'validation']:
        num_images = 15 if split == 'train' else 8
        for class_name in ['defect', 'no_defect']:
            path = os.path.join(DATASET_PATH, split, class_name)
            for i in range(num_images):
                img = np.random.randint(200, 250, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                for j in range(0, IMG_SIZE, 4):
                    cv2.line(img, (j, 0), (j, IMG_SIZE), (180, 180, 180), 1)
                    cv2.line(img, (0, j), (IMG_SIZE, j), (180, 180, 180), 1)
                if class_name == 'defect':
                    cv2.circle(img, (IMG_SIZE//2, IMG_SIZE//2), 30, (100, 100, 100), -1)
                cv2.imwrite(os.path.join(path, f'sample_{i}.jpg'), img)
    
    print("   ✅ Dataset created")

# Create data generators
print("\n📊 Loading dataset...")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
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

print(f"   Train: {train_generator.samples}, Val: {validation_generator.samples}")

# Build simple custom CNN (NO BatchNormalization issues!)
print("\n🏗️  Building simple CNN model...")

model = models.Sequential([
    # Input
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 4
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    
    # Dense layers
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    # Output
    layers.Dense(2, activation='softmax')
])

print("   ✅ Model created")
print(f"   Parameters: {model.count_params():,}")

# Compile
print("\n⚙️  Compiling...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
print(f"\n🚀 Training for {EPOCHS} epochs...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=1
)

# Save
MODEL_PATH = 'textile_defect_model.keras'
print(f"\n💾 Saving model to {MODEL_PATH}...")
model.save(MODEL_PATH)

print(f"   ✅ Saved ({os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB)")

# Verify
print(f"\n🔍 Verifying compatibility...")
try:
    test_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    test_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Test prediction
    dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype('float32')
    pred = test_model.predict(dummy, verbose=0)
    
    print(f"   ✅ Model is FULLY COMPATIBLE with TensorFlow {tf.__version__}")
    print(f"   ✅ Test prediction successful: {pred.shape}")
except Exception as e:
    print(f"   ❌ Error: {str(e)}")
    exit(1)

print("\n" + "="*70)
print("✅ SUCCESS! MODEL IS READY FOR WEB APP")
print("="*70)

print(f"\n📊 Training Results:")
print(f"   Final Train Accuracy: {history.history['accuracy'][-1]:.2%}")
print(f"   Final Val Accuracy: {history.history['val_accuracy'][-1]:.2%}")

print(f"\n🚀 Next Steps:")
print(f"   1. Model is saved: {MODEL_PATH}")
print(f"   2. Refresh your browser (app is already running)")
print(f"   3. Upload fabric images to test!")

print("\n💡 Note: This is a simple CNN for demo purposes.")
print("   For production, train longer with your real dataset.")

print("\n" + "="*70 + "\n")
