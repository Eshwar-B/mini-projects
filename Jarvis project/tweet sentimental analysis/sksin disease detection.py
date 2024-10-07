import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

# Data Paths
train_data_dir = '/path/to/your/train/data'  # Update with your path
test_data_dir = '/path/to/your/test/data'    # Update with your path

# Image Data Generators for Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(192, 192),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(192, 192),
    batch_size=32,
    class_mode='categorical'
)

# Class Information
class_names = sorted(os.listdir(train_data_dir))
num_classes = len(class_names)
print('Classes:', class_names)

# CNN Model Definition
cnn_model = Sequential()

# Input Layer
cnn_model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(192, 192, 3)))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Block 1
cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Block 2
cnn_model.add(Conv2D(256, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and Dense Layers
cnn_model.add(Flatten())
cnn_model.add(Dense(512, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(num_classes, activation='softmax'))

# Compile the Model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()

# Model Checkpoint
checkpoint_callback = ModelCheckpoint('skin_disease_model.h5', save_best_only=True)

# Train the Model
history = cnn_model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=[checkpoint_callback]
)

# Plot Training History
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
