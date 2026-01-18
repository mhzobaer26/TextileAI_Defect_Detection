"""
Retrain TextileAI Model with Proper Settings
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

print("🧵 Retraining TextileAI Model...")
print(f"TensorFlow Version: {tf.__version__}\n")

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001  # Lower learning rate
DATASET_PATH = 'Dataset'

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
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

print(f"✅ Train: {train_generator.samples} images")
print(f"✅ Validation: {validation_generator.samples} images")
print(f"✅ Classes: {train_generator.class_indices}\n")

# Build a simpler, more trainable model
print("🏗️  Building model...")

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    # Block 4
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    # Dense layers
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    
    # Output layer
    layers.Dense(2, activation='softmax')
])

print(f"✅ Model created ({model.count_params():,} parameters)\n")

# Compile with appropriate optimizer
print("⚙️  Compiling...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
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

print(f"\n🚀 Training for up to {EPOCHS} epochs...")
print("=" * 60)

# Train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 60)
print("✅ Training Complete!")
print("\n📊 Final Results:")
print(f"   Best Validation Accuracy: {max(history.history['val_accuracy']):.2%}")
print(f"   Best Validation Loss: {min(history.history['val_loss']):.4f}")

# Evaluate on validation set
print("\n🔍 Evaluating on validation set...")
val_loss, val_accuracy, val_precision, val_recall = model.evaluate(validation_generator, verbose=0)
print(f"   Validation Accuracy: {val_accuracy:.2%}")
print(f"   Validation Precision: {val_precision:.2%}")
print(f"   Validation Recall: {val_recall:.2%}")

print("\n💾 Model saved as: best_model.keras")
print("\n✅ Done! Refresh your browser to use the new model.")
