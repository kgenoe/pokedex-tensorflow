import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


############################################################
#
# This file was created following this guide:
# https://www.tensorflow.org/tutorials/images/classification
#
############################################################



# set path to data directory
import pathlib
data_dir = pathlib.Path("TestData30/")


## Load Images
batch_size = 32
img_height = 160
img_width = 160

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# output all class names found in the data
class_names = train_ds.class_names
num_classes = len(class_names)
print("Importing ", num_classes, " classes")
# print(class_names)


## Configure dataset for performance
# .cache() caches dataset in memory between epochs
# .prefetch() prefetches next elements in dataset while previous
#       is being processed by the model
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



# Create the model

model = Sequential([

  # Normalize image size
  # layers.experimental.preprocessing.Resizing(img_height, img_width),

  # Normalize RGB values [1-255] to between 0-1
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  
  # Data augmentation to expose more aspects of a limited data set (helps with overfit)
  layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
  layers.experimental.preprocessing.RandomRotation(0.2),
  layers.experimental.preprocessing.RandomZoom(0.1),
  
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  # Randomly drops % of output units during training,
  layers.Dropout(0.2),

  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#output summary of model
model.summary()




## Train the model
epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


## Output results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("Training Accuracy:")
print(['%.2f' % elem for elem in acc])

print("Validation Accuracy:")
print(['%.2f' % elem for elem in val_acc])

# print("\nTraining Loss:")
# print(['%.2f' % elem for elem in loss])

# print("Validation Loss:")
# print(['%.2f' % elem for elem in val_loss])
# print("")