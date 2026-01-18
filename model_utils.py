"""
Model Utilities for TextileAI Defect Detection
Contains functions for image preprocessing and prediction
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Configuration
IMG_SIZE = 224
CLASS_NAMES = ['defect', 'no_defect']

def preprocess_image(image):
    """
    Preprocess uploaded image for model prediction
    
    Args:
        image: PIL Image object
        
    Returns:
        numpy array: Preprocessed image ready for prediction
    """
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Ensure 3 channels (should be guaranteed by RGB conversion above)
    if len(img_array.shape) == 2:  # Grayscale (safety check)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA (safety check)
        img_array = img_array[:, :, :3]  # Drop alpha channel
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Ensure shape is (224, 224, 3)
    if img_normalized.shape != (IMG_SIZE, IMG_SIZE, 3):
        raise ValueError(f"Image shape mismatch: expected (224, 224, 3), got {img_normalized.shape}")
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch


def predict_defect(model, processed_image):
    """
    Make prediction using the trained model
    
    Args:
        model: Loaded Keras model
        processed_image: Preprocessed image array
        
    Returns:
        tuple: (prediction array, confidence score, class name)
    """
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)
    
    # Get predicted class index
    predicted_class_idx = np.argmax(prediction[0])
    
    # Get confidence score
    confidence = float(prediction[0][predicted_class_idx])
    
    # Get class name
    class_name = CLASS_NAMES[predicted_class_idx].replace('_', ' ').title()
    
    return prediction, confidence, class_name


def get_detailed_analysis(class_name, confidence):
    """
    Generate detailed analysis based on prediction
    
    Args:
        class_name: Predicted class name
        confidence: Confidence score
        
    Returns:
        dict: Detailed analysis including status and recommendations
    """
    analysis = {}
    
    if class_name == "No Defect":
        analysis['status'] = "PASS ✅"
        
        if confidence >= 0.95:
            analysis['confidence_level'] = "Very High"
            analysis['recommendation'] = """
            **Excellent Quality!** 
            
            The fabric shows no signs of defects with very high confidence. 
            
            ✅ Safe to proceed with:
            - Manufacturing processes
            - Quality approval
            - Shipping to customers
            """
        elif confidence >= 0.80:
            analysis['confidence_level'] = "High"
            analysis['recommendation'] = """
            **Good Quality**
            
            The fabric appears to be defect-free with high confidence.
            
            ✅ Recommended actions:
            - Approve for production
            - Continue with normal processing
            - Standard quality documentation
            """
        else:
            analysis['confidence_level'] = "Moderate"
            analysis['recommendation'] = """
            **Likely No Defect**
            
            The fabric appears to be defect-free, but confidence is moderate.
            
            ⚠️ Suggested actions:
            - Visual inspection recommended
            - Check image quality (lighting, focus)
            - Re-capture image if possible
            - Consider secondary inspection
            """
    
    else:  # Defect detected
        analysis['status'] = "FAIL ❌"
        
        if confidence >= 0.95:
            analysis['confidence_level'] = "Very High"
            analysis['recommendation'] = """
            **Defect Confirmed!**
            
            The model detected a defect with very high confidence.
            
            ❌ Immediate actions required:
            - Reject the fabric
            - Quarantine the batch
            - Notify quality control team
            - Investigate root cause
            - Document for supplier feedback
            """
        elif confidence >= 0.80:
            analysis['confidence_level'] = "High"
            analysis['recommendation'] = """
            **Defect Likely Present**
            
            A defect has been detected with high confidence.
            
            ❌ Required actions:
            - Manual inspection needed
            - Mark for review
            - Isolate from good batches
            - Document the issue
            - Consider rework if possible
            """
        else:
            analysis['confidence_level'] = "Moderate"
            analysis['recommendation'] = """
            **Possible Defect Detected**
            
            The model suggests a potential defect, but confidence is moderate.
            
            ⚠️ Recommended steps:
            - Careful manual inspection required
            - Verify with quality expert
            - Check image quality
            - Re-test with better lighting
            - Document findings
            """
    
    return analysis


def get_confidence_interpretation(confidence):
    """
    Interpret confidence score
    
    Args:
        confidence: Float confidence score (0-1)
        
    Returns:
        str: Interpretation of confidence level
    """
    if confidence >= 0.95:
        return "🟢 Very High - Model is very certain"
    elif confidence >= 0.80:
        return "🟢 High - Model is confident"
    elif confidence >= 0.65:
        return "🟡 Moderate - Some uncertainty"
    elif confidence >= 0.50:
        return "🟡 Low - Significant uncertainty"
    else:
        return "🔴 Very Low - Model is uncertain"


def get_quality_metrics(prediction):
    """
    Calculate additional quality metrics
    
    Args:
        prediction: Model prediction array
        
    Returns:
        dict: Quality metrics
    """
    defect_prob = float(prediction[0][0])
    no_defect_prob = float(prediction[0][1])
    
    # Calculate uncertainty
    uncertainty = 1 - max(defect_prob, no_defect_prob)
    
    # Calculate decision margin
    margin = abs(defect_prob - no_defect_prob)
    
    metrics = {
        'defect_probability': defect_prob,
        'no_defect_probability': no_defect_prob,
        'uncertainty': uncertainty,
        'decision_margin': margin,
        'is_confident': margin > 0.5  # Clear decision if margin > 50%
    }
    
    return metrics
