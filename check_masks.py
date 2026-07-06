import os
import pandas as pd

IMG_DIR  = r"C:\Users\himan\.cache\kagglehub\datasets\surajghuwalewala\ham1000-segmentation-and-classification\versions\2\images"
MASK_DIR = r"C:\Users\himan\.cache\kagglehub\datasets\surajghuwalewala\ham1000-segmentation-and-classification\versions\2\masks"
CSV      = r"C:\Users\himan\.cache\kagglehub\datasets\surajghuwalewala\ham1000-segmentation-and-classification\versions\2\GroundTruth.csv"

df = pd.read_csv(CSV)
print("CSV columns:", df.columns.tolist())
print("Sample image names:", df.iloc[:, 0].head(5).tolist())

masks = [f for f in os.listdir(MASK_DIR) if f.endswith('.png')]
images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
print(f"\nTotal images: {len(images)}")
print(f"Total masks:  {len(masks)}")
print(f"Sample mask:  {masks[0]}")
print(f"Sample image: {images[0]}")