"""
Train Model with Your Real Dataset
Production-quality training with more epochs
"""

print("\n🧵 Training TextileAI with YOUR Dataset...\n")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from datetime import datetime

print(f"TensorFlow Version: {tf.__version__}")

# Configuration - PRODUCTION SETTINGS
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30  # More epochs for better accuracy
LEARNING_RATE = 0.001

print("\n⚙️  Configuration:")
print(f"   Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS} (production training)")
print(f"   Learning Rate: {LEARNING_RATE}")

# Check dataset
DATASET_PATH = 'Dataset'
if not os.path.exists(DATASET_PATH):
    print("\n❌ Dataset folder not found!")
    print("   Please ensure Dataset/ folder exists with train/validation/test splits")
    exit(1)

# Create data generators
print("\n📊 Loading your dataset...")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'validation'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"   ✅ Train: {train_generator.samples} images")
print(f"   ✅ Validation: {validation_generator.samples} images")
print(f"   ✅ Classes: {list(train_generator.class_indices.keys())}")

# Build model
print("\n🏗️  Building CNN model...")

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 4
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    
    # Dense layers
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
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
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train
print(f"\n🚀 Training for up to {EPOCHS} epochs...")
print("   (This may take 10-30 minutes depending on dataset size)\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# Save final model
MODEL_PATH = 'textile_defect_model.keras'
print(f"\n💾 Saving final model...")
model.save(MODEL_PATH)
print(f"   ✅ Saved as {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB)")

# Verify
print(f"\n🔍 Verifying model...")
try:
    test_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    test_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype('float32')
    pred = test_model.predict(dummy, verbose=0)
    print(f"   ✅ Model verified and ready!")
except Exception as e:
    print(f"   ❌ Error: {str(e)}")
    exit(1)

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

print(f"\n📊 Final Results:")
print(f"   Best Train Accuracy: {max(history.history['accuracy']):.2%}")
print(f"   Best Val Accuracy: {max(history.history['val_accuracy']):.2%}")
print(f"   Final Train Accuracy: {history.history['accuracy'][-1]:.2%}")
print(f"   Final Val Accuracy: {history.history['val_accuracy'][-1]:.2%}")

if 'precision' in history.history:
    print(f"   Final Precision: {history.history['precision'][-1]:.2%}")
if 'recall' in history.history:
    print(f"   Final Recall: {history.history['recall'][-1]:.2%}")

print(f"\n🎯 Model Performance:")
val_acc = max(history.history['val_accuracy'])
if val_acc >= 0.95:
    print(f"   🌟 EXCELLENT! Validation accuracy: {val_acc:.2%}")
elif val_acc >= 0.85:
    print(f"   ✅ GOOD! Validation accuracy: {val_acc:.2%}")
elif val_acc >= 0.75:
    print(f"   ⚠️  ACCEPTABLE. Validation accuracy: {val_acc:.2%}")
else:
    print(f"   ⚠️  Consider training longer. Validation accuracy: {val_acc:.2%}")

print(f"\n🚀 Next Steps:")
print(f"   1. Model saved: {MODEL_PATH}")
print(f"   2. Refresh browser (app is running)")
print(f"   3. Upload your fabric images to test!")

print("\n💡 Tips for better accuracy:")
print("   - Add more training images (aim for 500+ per class)")
print("   - Ensure images are clear and well-lit")
print("   - Balance defect and no_defect images")
print("   - Train for more epochs if needed")

print("\n" + "="*70 + "\n")
