"""
Check model compatibility with current TensorFlow version
"""

import tensorflow as tf
import os
from pathlib import Path

print("="*60)
print("🔍 TextileAI - Model Compatibility Checker")
print("="*60)

# Check TensorFlow version
print(f"\n📦 Current TensorFlow Version: {tf.__version__}")

# Check for model file
model_path = 'textile_defect_model.keras'

if not os.path.exists(model_path):
    print(f"\n❌ Model file not found: {model_path}")
    print("\n💡 Next steps:")
    print("   1. Train your model in the notebook")
    print("   2. Run: python export_model.py")
    exit(1)

print(f"\n✅ Model file found: {model_path}")
print(f"   Size: {Path(model_path).stat().st_size / (1024*1024):.2f} MB")

# Try loading the model
print("\n🔄 Attempting to load model...")

try:
    # Try with compile=False first
    print("   Method 1: Loading with compile=False...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("   ✅ Model loaded successfully (without compilation)")
    
    # Try recompiling
    print("\n   Recompiling model...")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("   ✅ Model recompiled successfully")
    
    # Test prediction
    print("\n   Testing prediction with dummy data...")
    import numpy as np
    dummy_input = np.random.rand(1, 224, 224, 3).astype('float32')
    prediction = model.predict(dummy_input, verbose=0)
    print(f"   ✅ Prediction successful: {prediction.shape}")
    
    print("\n" + "="*60)
    print("✅ MODEL IS COMPATIBLE!")
    print("="*60)
    print("\n🚀 You can now run the web app:")
    print("   streamlit run app.py")
    
except Exception as e:
    print(f"\n   ❌ Error: {str(e)}")
    print("\n" + "="*60)
    print("⚠️  MODEL COMPATIBILITY ISSUE DETECTED")
    print("="*60)
    
    print("\n📋 Problem:")
    print(f"   Your model was likely trained with TensorFlow 2.15.0")
    print(f"   Current version is TensorFlow {tf.__version__}")
    print("   There are breaking changes in BatchNormalization layers")
    
    print("\n💡 Solutions:")
    print("\n   Option 1: Re-train the model (RECOMMENDED)")
    print("   - Open TextileAI_Defect_Detection.ipynb")
    print("   - Change TensorFlow version to 2.20.0 in first cell")
    print("   - Run all cells to retrain with new version")
    print("   - Run: python export_model.py")
    
    print("\n   Option 2: Use Python 3.10 or 3.11")
    print("   - Create new venv with Python 3.10/3.11")
    print("   - Install TensorFlow 2.15.0")
    print("   - Old model will work without issues")
    
    print("\n   Option 3: Convert model format")
    print("   - Load model in original environment")
    print("   - Save weights only and rebuild architecture")
    
    print("\n" + "="*60)
