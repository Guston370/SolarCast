import os
import numpy as np
import pandas as pd
import cv2
import random
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# config
IMG_SIZE = 128
MAX_IMAGES = 5000

# Detect environment and set dataset path
# Check if running in Kaggle Notebook (data already available in cloud)
if os.path.exists("/kaggle/input"):
    # Running in Kaggle Notebook - use cloud path directly (no download)
    DATA_DIR = "/kaggle/input/sky-images-and-solar-radiation-measurement-dataset"
    print("Running in Kaggle environment. Using cloud dataset path.")
else:
    # Running locally - download dataset from Kaggle
    import kagglehub
    print("Downloading dataset from Kaggle...")
    DATA_DIR = kagglehub.dataset_download("alenjakopli/sky-images-and-solar-radiation-measurement-dataset")
    print("Path to dataset files:", DATA_DIR)

# load image paths
image_paths = []

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            image_paths.append(os.path.join(root, file))

print("Total images found:", len(image_paths))

if len(image_paths) == 0:
    print("Error: No images found in the dataset.")
    print(f"Dataset path: {os.path.abspath(DATA_DIR)}")
    exit(1)

# Shuffle and take subset
random.shuffle(image_paths)
image_paths = image_paths[:MAX_IMAGES]

print("Using images:", len(image_paths))

# extract lable
def extract_irradiance(filename):
    try:
        value = filename.split("_")[-1].split(".")[0]
        return float(value)
    except:
        return None

# load images and labels
X = []
y = []

for path in tqdm(image_paths):
    img = cv2.imread(path)
    if img is None:
        continue
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    
    label = extract_irradiance(os.path.basename(path))
    
    if label is None:
        continue
    
    X.append(img)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Final dataset:", X.shape, y.shape)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# build CNN model
model = models.Sequential([
    
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.GlobalAveragePooling2D(),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(1)
])

# compile model
model.compile(
    optimizer='adam',
    loss='mse',     # because you're doing regression
    metrics=['mae']
)

# build model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# evaluation
preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("RMSE:", rmse)
print("R2 Score:", r2)

# visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(y_test, preds, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Irradiance")
plt.show()

residuals = y_test - preds.flatten()

plt.scatter(preds, residuals, alpha=0.5)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()
# Residuals show non-random patterns at higher irradiance levels, indicating model bias and reduced accuracy under high-intensity solar conditions.

plt.hist(residuals, bins=50)
plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()
# Error distribution is centered near zero but exhibits slight skewness, indicating minor systematic underestimation.

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

# The training and validation loss curves remain close, indicating controlled overfitting, though overall loss suggests moderate underfitting
