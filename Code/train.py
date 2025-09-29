import numpy as np
import pandas as pd
from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm.notebook import tqdm_notebook as tqdm
import os
import shutil
import random

# Dataset setup (fallback to synthetic data if real dataset is missing)
dataset_root = "real_and_fake_face"
real_dir = os.path.join(dataset_root, "training_real")
fake_dir = os.path.join(dataset_root, "training_fake")

IS_SYNTHETIC = False

if not (os.path.isdir(real_dir) and os.path.isdir(fake_dir)):
    IS_SYNTHETIC = True
    dataset_root = "synth_data"
    real_dir = os.path.join(dataset_root, "real")
    fake_dir = os.path.join(dataset_root, "fake")

    # Generate a small synthetic dataset if not already present
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    def generate_circle_image(image_size=(96, 96)):
        image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        center = (random.randint(24, 72), random.randint(24, 72))
        radius = random.randint(10, 30)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        cv2.circle(image, center, radius, color, -1)
        return image

    def generate_rectangle_image(image_size=(96, 96)):
        image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        x1, y1 = random.randint(5, 40), random.randint(5, 40)
        x2, y2 = random.randint(56, 90), random.randint(56, 90)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        # Add light noise to differentiate class
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        return image

    # Only generate if directory is empty (idempotent)
    if len(os.listdir(real_dir)) == 0 and len(os.listdir(fake_dir)) == 0:
        num_per_class = 400
        for i in range(num_per_class):
            cv2.imwrite(os.path.join(real_dir, f"real_{i:04d}.jpg"), generate_circle_image())
            cv2.imwrite(os.path.join(fake_dir, f"fake_{i:04d}.jpg"), generate_rectangle_image())

if not IS_SYNTHETIC:
    # Visualizing real and fake faces (skip for synthetic data)
    def load_img(path):
        image = cv2.imread(path)
        image = cv2.resize(image, (224, 224))
        return image[..., ::-1]

    real_path = os.listdir(real_dir)
    fake_path = os.listdir(fake_dir)

    fig = plt.figure(figsize=(10, 10))
    for i in range(min(16, len(real_path))):
        plt.subplot(4, 4, i + 1)
        plt.imshow(load_img(os.path.join(real_dir, real_path[i])), cmap='gray')
        plt.suptitle("Real faces", fontsize=20)
        plt.axis('off')
    plt.show()

    fig = plt.figure(figsize=(10, 10))
    for i in range(min(16, len(fake_path))):
        plt.subplot(4, 4, i + 1)
        plt.imshow(load_img(os.path.join(fake_dir, fake_path[i])), cmap='gray')
        plt.suptitle("Fake faces", fontsize=20)
        plt.title(fake_path[i][:4])
        plt.axis('off')
    plt.show()

# Data augmentation
dataset_path = dataset_root
data_with_aug = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=False,
                                   rescale=1./255,
                                   validation_split=0.2)
train = data_with_aug.flow_from_directory(dataset_path,
                                          class_mode="binary",
                                          target_size=(96, 96),
                                          batch_size=32,
                                          subset="training")
val = data_with_aug.flow_from_directory(dataset_path,
                                          class_mode="binary",
                                          target_size=(96, 96),
                                          batch_size=32,
                                          subset="validation")

# MobileNetV2 model
mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
tf.keras.backend.clear_session()

model = Sequential([mnet,
                    GlobalAveragePooling2D(),
                    Dense(512, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(128, activation="relu"),
                    Dropout(0.1),
                    Dense(2, activation="softmax")])

model.layers[0].trainable = False

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

# Callbacks
def scheduler(epoch):
    if epoch <= 2:
        return 0.001
    elif epoch > 2 and epoch <= 15:
        return 0.0001
    else:
        return 0.00001

lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)
hist = model.fit(train,
                 epochs=20,
                 callbacks=[lr_callbacks],
                 validation_data=val)

# Save model
model.save('deepfake_detection_model.h5')

# Visualizing accuracy and loss
epochs = 20
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.grid(True)
plt.legend(['Train', 'Validation'])
plt.style.use(['classic'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.grid(True)
plt.legend(['Train', 'Validation'], loc=4)
plt.style.use(['classic'])
plt.show()
