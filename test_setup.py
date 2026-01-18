"""
Test script to verify TextileAI web app setup
Run this before starting the application
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 10:
        print("   ✅ Python version OK (3.10+)")
        return True
    else:
        print("   ❌ Python 3.10+ required")
        return False

def check_files():
    """Check if all required files exist"""
    print("\n📁 Checking Files:")
    
    required_files = {
        'app.py': 'Main application',
        'model_utils.py': 'Model utilities',
        'requirements.txt': 'Dependencies',
        'README.md': 'Documentation',
    }
    
    all_present = True
    for file, desc in required_files.items():
        if Path(file).exists():
            print(f"   ✅ {file:20s} - {desc}")
        else:
            print(f"   ❌ {file:20s} - {desc} (MISSING)")
            all_present = False
    
    return all_present

def check_model():
    """Check if model file exists"""
    print("\n🤖 Checking Model:")
    
    model_file = 'textile_defect_model.keras'
    
    if Path(model_file).exists():
        size = Path(model_file).stat().st_size / (1024 * 1024)
        print(f"   ✅ {model_file} found ({size:.2f} MB)")
        return True
    else:
        print(f"   ❌ {model_file} NOT FOUND")
        print("   ℹ️  You need to:")
        print("      1. Train the model in the notebook")
        print("      2. Run: python export_model.py")
        return False

def check_dependencies():
    """Check if key dependencies are installed"""
    print("\n📦 Checking Dependencies:")
    
    packages = {
        'streamlit': 'Web framework',
        'tensorflow': 'Deep learning',
        'cv2': 'Image processing (opencv)',
        'PIL': 'Image library (Pillow)',
        'numpy': 'Numerical computing'
    }
    
    all_installed = True
    
    for package, desc in packages.items():
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"   ✅ {package:15s} - {desc}")
        except ImportError:
            print(f"   ❌ {package:15s} - {desc} (NOT INSTALLED)")
            all_installed = False
    
    if not all_installed:
        print("\n   💡 To install missing packages:")
        print("      pip install -r requirements.txt")
    
    return all_installed

def run_checks():
    """Run all checks"""
    print("="*60)
    print("🧵 TextileAI - Setup Verification")
    print("="*60)
    
    results = []
    
    # Check Python version
    results.append(("Python Version", check_python_version()))
    
    # Check files
    results.append(("Required Files", check_files()))
    
    # Check model
    results.append(("Model File", check_model()))
    
    # Check dependencies
    results.append(("Dependencies", check_dependencies()))
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check_name:20s} : {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\n🚀 You're ready to run the app:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\n📝 Fix the issues above and run this script again.")
        print("\n💡 Quick fixes:")
        print("   1. Install packages: pip install -r requirements.txt")
        print("   2. Export model: python export_model.py")
    
    print("="*60)

if __name__ == "__main__":
    run_checks()
