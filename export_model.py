"""
Export trained model for web application
Run this script after training your model in the notebook
"""

import os
import shutil
from pathlib import Path

def find_latest_model():
    """Find the most recent .keras model file"""
    current_dir = Path('.')
    keras_files = list(current_dir.glob('*.keras'))
    
    if not keras_files:
        print("❌ No .keras model files found in current directory!")
        print("\nℹ️  Please make sure you have trained the model first.")
        print("   The model file should end with .keras extension")
        return None
    
    # Sort by modification time, get the latest
    latest_model = max(keras_files, key=lambda p: p.stat().st_mtime)
    return latest_model

def export_model():
    """Export the trained model for web app"""
    print("🔍 Searching for trained model...\n")
    
    # Find latest model
    model_path = find_latest_model()
    
    if model_path is None:
        return
    
    print(f"✅ Found model: {model_path.name}")
    print(f"   Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Target filename for web app
    target_name = 'textile_defect_model.keras'
    
    # Check if target already exists
    if Path(target_name).exists():
        print(f"\n⚠️  {target_name} already exists!")
        response = input("   Do you want to replace it? (yes/no): ").lower().strip()
        
        if response not in ['yes', 'y']:
            print("\n❌ Export cancelled.")
            return
    
    # Copy the model
    try:
        shutil.copy2(model_path, target_name)
        print(f"\n✅ Model exported successfully!")
        print(f"   Saved as: {target_name}")
        
        # Verify the model can be loaded
        print(f"\n🔍 Verifying model compatibility...")
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(target_name, compile=False)
            print(f"   ✅ Model verified - compatible with TensorFlow {tf.__version__}")
        except Exception as e:
            print(f"   ⚠️  Warning: {str(e)}")
            print(f"   You may need to re-train with current TensorFlow version")
        
        print(f"\n🚀 You can now run the web app:")
        print(f"   streamlit run app.py")
    except Exception as e:
        print(f"\n❌ Error exporting model: {str(e)}")

def verify_files():
    """Verify all required files are present"""
    print("\n" + "="*50)
    print("📋 Checking required files for web app...")
    print("="*50 + "\n")
    
    required_files = {
        'app.py': 'Main web application',
        'model_utils.py': 'Model utilities',
        'requirements.txt': 'Python dependencies',
        'textile_defect_model.keras': 'Trained model',
        'README.md': 'Documentation'
    }
    
    all_present = True
    
    for file, description in required_files.items():
        if Path(file).exists():
            size = Path(file).stat().st_size / 1024  # KB
            print(f"✅ {file:30s} - {description:30s} ({size:.2f} KB)")
        else:
            print(f"❌ {file:30s} - {description:30s} (MISSING)")
            all_present = False
    
    print("\n" + "="*50)
    
    if all_present:
        print("✅ All files present! Ready to run web app!")
        print("\n🚀 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Run the app: streamlit run app.py")
    else:
        print("⚠️  Some files are missing. Please check above.")
        print("   Make sure you've trained the model and exported it.")
    
    print("="*50)

if __name__ == "__main__":
    print("="*50)
    print("🧵 TextileAI - Model Export Tool")
    print("="*50 + "\n")
    
    export_model()
    verify_files()
    
    print("\n✨ Export process completed!")
