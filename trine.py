import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json
from sklearn.utils.class_weight import compute_class_weight

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

print("🚀 Starting Plant Disease Detection with Transfer Learning...")

# Paths
train_dir = r"PlantVillage\train"
val_dir = r"PlantVillage\val"

# Memory-efficient settings
img_height, img_width = 128, 128
batch_size = 16

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

print("📁 Loading datasets...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)

# Use pre-trained MobileNetV2
print("🧠 Loading pre-trained MobileNetV2...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)
base_model.trainable = False  # Freeze base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("📊 Model summary:")
model.summary()

# Train
print("🎯 Starting training...")
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=val_generator,
    callbacks=[
        EarlyStopping(patience=8, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ],
    verbose=1
)

model.save('plant_disease_model_mobilenet.h5')
print("💾 Model saved successfully!")