# 🧵 TextileAI Defect Detection - Web Application

A beautiful and intelligent web application for detecting defects in textile fabrics using AI-powered deep learning.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **🤖 AI-Powered Detection**: Uses EfficientNetB0 transfer learning model
- **📤 Easy Upload**: Drag-and-drop or browse to upload fabric images
- **📊 Detailed Analysis**: Confidence scores, probability breakdown, and recommendations
- **🎨 Beautiful UI**: Modern, responsive design with gradient cards and visual indicators
- **⚡ Real-time Processing**: Fast prediction and instant results
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices

## 📁 Project Structure

```
TextileAI_Defect_Detection/
│
├── 📱 FRONTEND & BACKEND
│   ├── app.py                              # Main Streamlit application
│   ├── model_utils.py                      # Model utilities and preprocessing
│   └── textile_defect_model.keras          # Trained model (you need to add this)
│
├── 📊 DATASET
│   └── Dataset/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── 📓 TRAINING
│   └── TextileAI_Defect_Detection.ipynb    # Model training notebook
│
├── 📋 CONFIGURATION
│   ├── requirements.txt                     # Python dependencies
│   ├── export_model.py                      # Script to export trained model
│   └── README.md                            # This file
│
└── 🚀 DEPLOYMENT
    └── (Instructions below)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Trained model file (`textile_defect_model.keras`)

### Step 1: Train Your Model First

1. Open `TextileAI_Defect_Detection.ipynb` in Jupyter or Google Colab
2. Run all cells to train the model
3. The model will be saved as `.keras` file
4. Run the export script to prepare the model:
   ```bash
   python export_model.py
   ```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 3: Run the Web Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 How to Use

1. **Upload Image**: Click "Browse files" or drag & drop a fabric image (JPG, PNG, JPEG)
2. **Wait for Analysis**: The AI model will process your image (takes 1-3 seconds)
3. **View Results**:
   - ✅ **No Defect**: Fabric is in good condition
   - ❌ **Defect**: Fabric has quality issues
4. **Check Details**: View confidence scores, probability breakdown, and recommendations
5. **Take Action**: Follow the recommendations based on the detection results

## 🎯 Model Information

- **Architecture**: EfficientNetB0 (Transfer Learning)
- **Framework**: TensorFlow/Keras
- **Input Size**: 224x224 pixels
- **Classes**: 
  - Defect
  - No Defect
- **Training**: Trained on synthetic textile dataset
- **Augmentation**: Rotation, shift, zoom, flip

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Easiest) ⭐

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Deploy your app
5. **Important**: Make sure `textile_defect_model.keras` is in your repo (or use Git LFS for large files)

### Option 2: Hugging Face Spaces

1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space (select Streamlit)
3. Upload files:
   - `app.py`
   - `model_utils.py`
   - `requirements.txt`
   - `textile_defect_model.keras`
4. Space will auto-deploy

### Option 3: Render

1. Create account at [render.com](https://render.com)
2. Create new Web Service
3. Connect GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

### Option 4: Local Network Sharing

```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

Access from other devices on your network at `http://YOUR_IP:8501`

## 📦 Model File Setup

The trained model file (`textile_defect_model.keras`) is **required** but not included in this repository due to size.

**To get the model:**

1. **Train it yourself**: 
   - Run `TextileAI_Defect_Detection.ipynb`
   - Model will be saved automatically
   
2. **Use existing model**:
   - Copy your trained `.keras` file
   - Rename to `textile_defect_model.keras`
   - Place in the project root directory

3. **Export from Colab**:
   ```python
   # Run this in your notebook after training
   from google.colab import files
   files.download('textile_defect_model_final.keras')
   ```

## 🔧 Configuration

### Change Model Path

Edit `app.py` line 56:
```python
model_path = 'textile_defect_model.keras'  # Change this path
```

### Adjust Image Size

Edit `model_utils.py` line 11:
```python
IMG_SIZE = 224  # Change if your model uses different size
```

### Modify Classes

Edit `model_utils.py` line 12:
```python
CLASS_NAMES = ['defect', 'no_defect']  # Update class names
```

## 📊 API Usage (Optional)

You can also use the model utilities as a Python module:

```python
from model_utils import preprocess_image, predict_defect
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('textile_defect_model.keras')

# Load and process image
image = Image.open('fabric.jpg')
processed = preprocess_image(image)

# Make prediction
prediction, confidence, class_name = predict_defect(model, processed)

print(f"Result: {class_name}")
print(f"Confidence: {confidence:.2%}")
```

## 🐛 Troubleshooting

### Model Not Found Error
- Ensure `textile_defect_model.keras` is in the same directory as `app.py`
- Check the model path in `app.py`

### Import Errors
- Run `pip install -r requirements.txt`
- Make sure you're using Python 3.10+

### Low Memory / Slow Performance
- Reduce batch size in model inference
- Use CPU instead of GPU for small-scale deployment
- Optimize image preprocessing

### Image Upload Issues
- Supported formats: JPG, JPEG, PNG
- Maximum file size: Check your Streamlit config
- Image should be clear and well-lit

## 📈 Future Enhancements

- [ ] Multi-defect type classification
- [ ] Defect localization (bounding boxes)
- [ ] Batch image processing
- [ ] Export analysis reports (PDF/CSV)
- [ ] Model performance monitoring
- [ ] User feedback collection
- [ ] Historical analysis dashboard
- [ ] REST API endpoint

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Created with ❤️ for textile quality control automation

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Streamlit for the amazing web framework
- EfficientNet authors for the model architecture

## 📞 Support

For issues, questions, or contributions, please create an issue in the repository.

---

**⭐ Star this repo if you find it helpful!**
