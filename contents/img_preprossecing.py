#library imports
import numpy as np
import tensorflow
import keras
import os
import glob
import skimage
from skimage import io
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#importing image dataset

data_dir = r"D:\ml-project\dataset\fruits-360_3-body-problem\fruits-360-3-body-problem"
train_dir = r"D:\ml-project\dataset\fruits-360_3-body-problem\fruits-360-3-body-problem\Training"
test_dir  = r"D:\ml-project\dataset\fruits-360_3-body-problem\fruits-360-3-body-problem\Test"

#step 1 Normalization of images

datagen = ImageDataGenerator(rescale=1./255)  # normalizes to [0,1]
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)
#checking normalization
x_batch, y_batch = next(train_data)
print("Batch min:", np.min(x_batch))
print("Batch max:", np.max(x_batch))
print("Batch mean:", np.mean(x_batch))
x_batch, _ = next(train_data)
print("Resized image shape:", x_batch[0].shape)


#step 2 Data Augmentation

aug_datagen = ImageDataGenerator(
    rescale=1./255 ,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#checking data augmentation
augmented_data = aug_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)   
x_aug_batch, y_aug_batch = next(augmented_data)
print("Augmented batch shape:", x_aug_batch.shape)
print("Augmented batch min:", np.min(x_aug_batch))
print("Augmented batch max:", np.max(x_aug_batch))


# Step 3: Feature Extraction 


from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray

def extract_features(img):
    """Extract combined features: color histogram + LBP + HOG"""
    features = []

    # 1. Color Histogram (for each RGB channel)
    for i in range(3):
        hist, _ = np.histogram(img[:,:,i], bins=32, range=(0,1))
        hist = hist / np.sum(hist)  # normalize
        features.extend(hist)

    # 2. Convert to grayscale for LBP and HOG
    gray = rgb2gray(img)
    gray_int = (gray * 255).astype(np.uint8)  # convert to uint8 for LBP

    # 3. Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray_int, P=8, R=1, method='uniform')  # use gray_int here
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    lbp_hist = lbp_hist / np.sum(lbp_hist)
    features.extend(lbp_hist)

    # 4. HOG features (can stay in float)
    hog_features = hog(gray, pixels_per_cell=(8,8), cells_per_block=(2,2), 
                       orientations=9, block_norm='L2-Hys', visualize=False)
    features.extend(hog_features)

    return np.array(features)

# Example: extract features from a batch of augmented images
feature_list = []
labels_list = []

for i in range(len(x_aug_batch)):
    img = x_aug_batch[i]
    features = extract_features(img)
    feature_list.append(features)
    labels_list.append(np.argmax(y_aug_batch[i]))  # convert one-hot to class index

# Convert to numpy arrays
X = np.array(feature_list)
y = np.array(labels_list)

print("Feature vector shape:", X.shape)
print("Labels shape:", y.shape)
  

     
    