import os

MASK_DIR = r"C:\Users\himan\Downloads\darmasalic\siim-isic-melanoma-classification\masks\ISIC2018_Task1_Training_GroundTruth"

mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith('_segmentation.png')]
print(f"Found {len(mask_files)} mask files")
print(f"Example: {mask_files[0]}")