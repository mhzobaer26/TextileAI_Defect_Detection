"""
Complete Setup and Issue Resolution Script
Run this to check all requirements and fix issues
"""

import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_python_version():
    """Check Python version"""
    print_header("🐍 STEP 1: Python Version Check")
    version = sys.version_info
    print(f"Current Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 10:
        print("✅ Python version is compatible (3.10+)")
        return True, version
    else:
        print("❌ Python 3.10+ required")
        return False, version

def check_dependencies():
    """Check and report on dependencies"""
    print_header("📦 STEP 2: Dependencies Check")
    
    required_packages = {
        'streamlit': 'Web framework',
        'tensorflow': 'Deep learning',
        'cv2': 'Image processing',
        'PIL': 'Image library',
        'numpy': 'Numerical computing'
    }
    
    missing = []
    installed = []
    
    for package, desc in required_packages.items():
        try:
            if package == 'cv2':
                __import__('cv2')
                import cv2
                installed.append(f"✅ opencv-python: {cv2.__version__}")
            elif package == 'PIL':
                from PIL import Image
                installed.append(f"✅ Pillow: {Image.__version__ if hasattr(Image, '__version__') else 'installed'}")
            else:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
                installed.append(f"✅ {package}: {version}")
        except ImportError:
            missing.append(f"❌ {package} - {desc}")
    
    for item in installed:
        print(f"   {item}")
    
    if missing:
        print("\n   Missing packages:")
        for item in missing:
            print(f"   {item}")
        return False
    
    return True

def check_files():
    """Check required files"""
    print_header("📁 STEP 3: Required Files Check")
    
    required = {
        'app.py': 'Main web application',
        'model_utils.py': 'Model utilities',
        'requirements.txt': 'Dependencies list',
        'TextileAI_Defect_Detection.ipynb': 'Training notebook'
    }
    
    all_present = True
    for file, desc in required.items():
        if Path(file).exists():
            print(f"   ✅ {file:40s} - {desc}")
        else:
            print(f"   ❌ {file:40s} - {desc}")
            all_present = False
    
    return all_present

def check_model():
    """Check for model file"""
    print_header("🤖 STEP 4: Model File Check")
    
    model_file = 'textile_defect_model.keras'
    
    if Path(model_file).exists():
        size = Path(model_file).stat().st_size / (1024 * 1024)
        print(f"   ✅ Model found: {model_file} ({size:.2f} MB)")
        
        # Check compatibility
        try:
            import tensorflow as tf
            print(f"   🔍 Testing with TensorFlow {tf.__version__}...")
            model = tf.keras.models.load_model(model_file, compile=False)
            print(f"   ✅ Model is compatible!")
            return True, "compatible"
        except Exception as e:
            print(f"   ⚠️  Compatibility issue: {str(e)[:100]}...")
            return True, "incompatible"
    else:
        print(f"   ❌ Model not found: {model_file}")
        print(f"   📝 You need to train the model first")
        return False, "missing"

def check_dataset():
    """Check dataset structure"""
    print_header("📊 STEP 5: Dataset Check")
    
    if not Path('Dataset').exists():
        print("   ❌ Dataset folder not found")
        return False
    
    required_structure = [
        'Dataset/train/defect',
        'Dataset/train/no_defect',
        'Dataset/validation/defect',
        'Dataset/validation/no_defect',
        'Dataset/test/defect',
        'Dataset/test/no_defect',
    ]
    
    all_present = True
    for path in required_structure:
        if Path(path).exists():
            count = len(list(Path(path).glob('*.*')))
            print(f"   ✅ {path:40s} ({count} images)")
        else:
            print(f"   ❌ {path:40s} (missing)")
            all_present = False
    
    return all_present

def provide_solutions(results):
    """Provide solutions based on checks"""
    print_header("💡 RECOMMENDED ACTIONS")
    
    python_ok, python_version = results['python']
    deps_ok = results['dependencies']
    files_ok = results['files']
    model_status, model_state = results['model']
    dataset_ok = results['dataset']
    
    actions = []
    
    if not deps_ok:
        actions.append({
            'priority': 'HIGH',
            'action': 'Install Dependencies',
            'command': 'pip install -r requirements.txt',
            'description': 'Required packages are missing'
        })
    
    if model_state == "missing":
        actions.append({
            'priority': 'HIGH',
            'action': 'Train Model',
            'command': 'Open TextileAI_Defect_Detection.ipynb and run all cells',
            'description': 'No trained model found - train first'
        })
    elif model_state == "incompatible":
        actions.append({
            'priority': 'CRITICAL',
            'action': 'Re-train Model',
            'command': 'Re-run notebook with updated TensorFlow version',
            'description': 'Model was trained with different TensorFlow version'
        })
    
    if model_status and model_state == "compatible" and deps_ok:
        actions.append({
            'priority': 'READY',
            'action': '🚀 Run Web App',
            'command': 'streamlit run app.py',
            'description': 'Everything is ready!'
        })
    
    if not dataset_ok:
        actions.append({
            'priority': 'MEDIUM',
            'action': 'Check Dataset',
            'command': 'Ensure Dataset folder has train/val/test splits',
            'description': 'Dataset structure incomplete'
        })
    
    # Print actions by priority
    for priority in ['CRITICAL', 'HIGH', 'READY', 'MEDIUM', 'LOW']:
        priority_actions = [a for a in actions if a['priority'] == priority]
        if priority_actions:
            for action in priority_actions:
                if priority == 'CRITICAL':
                    icon = '🚨'
                elif priority == 'HIGH':
                    icon = '⚠️'
                elif priority == 'READY':
                    icon = '✅'
                else:
                    icon = '📌'
                
                print(f"\n{icon} [{action['priority']}] {action['action']}")
                print(f"   Description: {action['description']}")
                print(f"   Command: {action['command']}")

def main():
    """Main setup check"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║        🧵 TextileAI - Complete Setup & Issue Resolution          ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # Run all checks
    results['python'] = check_python_version()
    results['dependencies'] = check_dependencies()
    results['files'] = check_files()
    results['model'] = check_model()
    results['dataset'] = check_dataset()
    
    # Provide solutions
    provide_solutions(results)
    
    # Summary
    print_header("📋 SUMMARY")
    
    python_ok, _ = results['python']
    deps_ok = results['dependencies']
    files_ok = results['files']
    model_status, model_state = results['model']
    dataset_ok = results['dataset']
    
    total_checks = 5
    passed_checks = sum([
        python_ok,
        deps_ok,
        files_ok,
        model_status,
        dataset_ok
    ])
    
    print(f"\n   Checks Passed: {passed_checks}/{total_checks}")
    
    if passed_checks == total_checks and model_state == "compatible":
        print("\n   🎉 ALL SYSTEMS GO!")
        print("   Your TextileAI application is ready to run!")
        print("\n   Next: streamlit run app.py")
    elif model_state == "incompatible":
        print("\n   ⚠️  MODEL COMPATIBILITY ISSUE")
        print("   Please re-train your model with the current TensorFlow version")
        print("   Steps:")
        print("   1. Open TextileAI_Defect_Detection.ipynb")
        print("   2. Run all cells (TensorFlow 2.20.0 will be installed)")
        print("   3. Run: python export_model.py")
        print("   4. Run: streamlit run app.py")
    else:
        print("\n   ⚠️  SETUP INCOMPLETE")
        print("   Please follow the recommended actions above")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
