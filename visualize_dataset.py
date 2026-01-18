"""
Visualize Dataset Images to Check Visual Differences
"""
from PIL import Image
import os
import numpy as np

def analyze_image_pair():
    """Compare a defect and no_defect image"""
    
    # Get first image from each class
    defect_path = os.path.join('Dataset/train/defect', os.listdir('Dataset/train/defect')[0])
    nodefect_path = os.path.join('Dataset/train/no_defect', os.listdir('Dataset/train/no_defect')[0])
    
    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    
    # Load images
    defect_img = Image.open(defect_path)
    nodefect_img = Image.open(nodefect_path)
    
    # Convert to arrays
    defect_arr = np.array(defect_img)
    nodefect_arr = np.array(nodefect_img)
    
    print(f"\nDEFECT IMAGE: {os.path.basename(defect_path)}")
    print(f"  Mode: {defect_img.mode}")
    print(f"  Size: {defect_img.size}")
    print(f"  Pixel stats: min={defect_arr.min()}, max={defect_arr.max()}, mean={defect_arr.mean():.1f}, std={defect_arr.std():.1f}")
    
    print(f"\nNO DEFECT IMAGE: {os.path.basename(nodefect_path)}")
    print(f"  Mode: {nodefect_img.mode}")
    print(f"  Size: {nodefect_img.size}")
    print(f"  Pixel stats: min={nodefect_arr.min()}, max={nodefect_arr.max()}, mean={nodefect_arr.mean():.1f}, std={nodefect_arr.std():.1f}")
    
    # Calculate difference
    if defect_arr.shape == nodefect_arr.shape:
        diff = np.abs(defect_arr.astype(float) - nodefect_arr.astype(float))
        print(f"\nDIFFERENCE BETWEEN IMAGES:")
        print(f"  Mean absolute difference: {diff.mean():.1f}")
        print(f"  Max difference: {diff.max():.1f}")
        print(f"  Std of difference: {diff.std():.1f}")
    
    # Check all images statistics
    print("\n" + "=" * 70)
    print("CHECKING ALL IMAGES...")
    print("=" * 70)
    
    defect_means = []
    defect_stds = []
    for img_name in os.listdir('Dataset/train/defect'):
        img = np.array(Image.open(os.path.join('Dataset/train/defect', img_name)))
        defect_means.append(img.mean())
        defect_stds.append(img.std())
    
    nodefect_means = []
    nodefect_stds = []
    for img_name in os.listdir('Dataset/train/no_defect'):
        img = np.array(Image.open(os.path.join('Dataset/train/no_defect', img_name)))
        nodefect_means.append(img.mean())
        nodefect_stds.append(img.std())
    
    print(f"\nDEFECT IMAGES (n={len(defect_means)}):")
    print(f"  Mean pixel value: {np.mean(defect_means):.1f} +/- {np.std(defect_means):.1f}")
    print(f"  Std pixel value: {np.mean(defect_stds):.1f} +/- {np.std(defect_stds):.1f}")
    
    print(f"\nNO DEFECT IMAGES (n={len(nodefect_means)}):")
    print(f"  Mean pixel value: {np.mean(nodefect_means):.1f} +/- {np.std(nodefect_means):.1f}")
    print(f"  Std pixel value: {np.mean(nodefect_stds):.1f} +/- {np.std(nodefect_stds):.1f}")
    
    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    mean_diff = abs(np.mean(defect_means) - np.mean(nodefect_means))
    
    if mean_diff < 5:
        print("\nWARNING: Images from both classes have very similar pixel statistics!")
        print("The mean pixel values differ by less than 5.")
        print("This makes it extremely difficult for the model to learn.")
        print("\nPossible issues:")
        print("  1. Defects are very subtle and hard to detect")
        print("  2. Images might need preprocessing (contrast enhancement, etc.)")
        print("  3. Dataset might be mislabeled")
        print("  4. Need more training data with clearer defects")
    else:
        print(f"\nGood: Classes have distinguishable statistics (diff: {mean_diff:.1f})")
    
    # Save sample images for manual inspection
    print("\n" + "=" * 70)
    print(f"Please manually inspect these images:")
    print(f"  Defect: {defect_path}")
    print(f"  No defect: {nodefect_path}")
    print("Check if you can visually see the difference!")
    print("=" * 70)

if __name__ == "__main__":
    analyze_image_pair()
