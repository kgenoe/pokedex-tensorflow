import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import coremltools as ct

############################################################
#
# This file was created following this guide:
# https://www.tensorflow.org/tutorials/images/classification
#
############################################################



# set path to data directory
import pathlib
data_name = "DataAll"
data_dir_name = data_name + "/"
data_dir = pathlib.Path(data_dir_name)


## Load Images
batch_size = 32
img_size = 160

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_size, img_size),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_size, img_size),
  batch_size=batch_size)


# output all class names found in the data
class_names = train_ds.class_names
num_classes = len(class_names)
print("Importing ", num_classes, " classes")


## Configure dataset for performance
# .cache() caches dataset in memory between epochs
# .prefetch() prefetches next elements in dataset while previous
#       is being processed by the model
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



## Create the model
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

resize = tf.keras.layers.experimental.preprocessing.Resizing(img_size, img_size)

rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

# Create the base model from the pre-trained model MobileNet V2
img_shape = (img_size, img_size, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = tf.keras.layers.Dense(num_classes)

inputs = tf.keras.Input(shape=img_shape)
x = resize(inputs)
x = rescale(x)
x = data_augmentation(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


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


# save keras model
model_name = 'pokedex-' + data_name
h5_model_name = model_name + '.h5'
model.save(h5_model_name) 

# Output model as coreml
image_input = [ct.ImageType(shape=(1, img_size, img_size, 3,),)]
classifier_config = ct.ClassifierConfig(class_names)
mlmodel = ct.convert(model,
                    inputs=image_input,
                    classifier_config=classifier_config)


## test coreml model on a test image 
# from PIL import Image
# test_image = Image.open("TestImages/1.png").resize((img_shape, img_shape))
# out_dict = mlmodel.predict({"input_2": test_image})
# print(out_dict)

ml_model_name = model_name + '.mlmodel'
mlmodel.save(ml_model_name)