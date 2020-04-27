#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Education - Udacity "Intro to TensorFlow for Deep Learning"
# Module: Flower Classifier Exercise
# Ref: https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c03_exercise_flowers_with_data_augmentation.ipynb#scrollTo=G1ymuCPS0_eu


# In[1]:


# Import packages.
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import glob
import shutil
import matplotlib.pyplot as plt
import numpy as np
import logging

# Using version 2.x of Tensorflow.
try:
  # Use the %tensorflow_version magic if in colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass

# Tensorflow.
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set logging level.
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Print Tensorflow version.
print('TensorFlow Version:', tf.__version__)


# In[2]:


# Download dataset.
print('Download dataset....')
data_url  = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
file_zip  = tf.keras.utils.get_file('flower_photos.tgz', origin=data_url, extract=True)
print('Download dataset completed.')


# In[3]:


# List directories.
dir_root  = os.path.join(os.path.dirname(file_zip))
dir_base  = os.path.join(dir_root, 'flower_photos')
dir_train = os.path.join(dir_base, 'train')
dir_val   = os.path.join(dir_base, 'validate')
classes   = [ 'daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

print('-----------------------')
print('- List Root Directory -')
print('-----------------------')
get_ipython().system('ls -lh $dir_root')
print('-----------------------')
print('-----------------------')
print('- List Base Directory -')
print('-----------------------')
get_ipython().system('ls -lh $dir_base')
print('-----------------------')


# In[4]:


# Moving data into respective classes.
print('Prepare data and directories....')
for i in classes:
  path_img    = os.path.join(dir_base, i)
  path_train  = os.path.join(dir_train, i)
  path_val    = os.path.join(dir_val, i)
  images      = glob.glob(path_img + '/*.jpg')

  print('  Process {}: {} images'.format(i, len(images)))
  if len(images) == 0:
    print('  No image available, skip the process.')
    continue

  # Seperate 80% to train and 20% to validation.
  train     = images[:round(len(images) * 0.8)] # [0, 80]
  validate  = images[round(len(images) * 0.8):] # [80, 100]

  print('    Train data:   ', len(train))
  print('    Validate data:', len(validate))

  # Move into respective directory, create directory if not exists.
  for t in train:
    if not os.path.exists(path_train):
      os.makedirs(path_train)
    shutil.move(t, path_train)

  for v in validate:
    if not os.path.exists(path_val):
      os.makedirs(path_val)
    shutil.move(v, path_val)
print('Prepare data completed.')


# In[5]:


print('-- Summary --')
print('Classes:       ', classes)
print('Dir Root:      ', dir_root)
print('Dir Base:      ', dir_base)
print('Dir Train:     ', dir_train)

for i in classes:
  path    = os.path.join(dir_train, i)
  images  = os.listdir(path)
  print('  {} ({}) images'.format(len(images), i))

print('Dir Validate:  ', dir_val)
for i in classes:
  path    = os.path.join(dir_val, i)
  images  = os.listdir(path)
  print('  {} ({}) images'.format(len(images), i))


# In[6]:


# Apply augmentation.
BATCH_SIZE = 100  # Number of training examples to process before updating our models variables.
IMG_SHAPE  = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels.

train_image_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    zoom_range=0.5,
    horizontal_flip=True,
    width_shift_range=0.15,
    height_shift_range=0.15
)
train_data_gen = train_image_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=dir_train,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='sparse'
)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images):
  fig, axes = plt.subplots(1, 5, figsize=(20, 20))
  axes      = axes.flatten()
  for img, ax in zip( images, axes):
    ax.imshow(img)
  plt.tight_layout()
  plt.show()

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


# In[7]:


# Create validate set.
val_image_gen = ImageDataGenerator(rescale=1./255)
val_data_gen = val_image_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=dir_val,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='sparse'
)


# In[ ]:


# Build the model.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# The exercise suggestion solution.
model_suggestiong = tf.keras.models.Sequential([
    # Why use padding='same'?
    tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),

    # Why don't add dropout after each 'max pooling' layer?

    # Why add dropout after flatten?
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),

    # Why add dropout here?
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])


# In[ ]:


# Compile the model.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[10]:


# Print summary.
model.summary()


# In[23]:


# Summary.
EPOCHS = 80
EPOCH_STEP_TRAIN = int(np.ceil(train_data_gen.n / BATCH_SIZE))
EPOCH_STEP_VALIDATE = int(np.ceil(val_data_gen.n / BATCH_SIZE))
print('-- Summary Data --')
print('Total Train Data:      ', train_data_gen.n)
print('Total Validataion Data:', val_data_gen.n)
print('Total Epochs:          ', EPOCHS)
print('Epoch Step (Train):    ', EPOCH_STEP_TRAIN)
print('Epoch Step (Validate): ', EPOCH_STEP_VALIDATE)


# In[24]:


# Train the model.
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=EPOCH_STEP_TRAIN,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=EPOCH_STEP_VALIDATE
)


# In[25]:


# Visualize the result.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Based on our model, we should probably stop the training after 30 epochs.

# In[47]:


# Get the dataset.
image_batch, label_batch = val_data_gen[0]

# Make predictions.
total = 20
fails = 0
plt.figure(figsize=(25, 25))
for i in range(total):
  plt.subplot(5, 4, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)

  img = image_batch[i]
  plt.imshow(img)

  img = np.array([img])
  predict_image = model.predict(img)
  predict_id    = np.argmax(predict_image[0])
  label_id      = int(label_batch[i])
  label_predict = classes[predict_id]
  label_correct = classes[label_id]
  color = 'blue'
  if not label_id == predict_id:
    fails = fails + 1
    color = 'red'
  plt.xlabel("{} ({}) ({}%)".format(label_predict, predict_id, 100 * np.max(predict_image)), color = color)
  plt.ylabel("{} ({})".format(label_correct, label_id))

print('Classes:', classes)
print('Fails:   {}%'.format(100 * (fails/total)))
plt.show()


# Our model has an accuracy about 70%, so failing about 30% is correct.
