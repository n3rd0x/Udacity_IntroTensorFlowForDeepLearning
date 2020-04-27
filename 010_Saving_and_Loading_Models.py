#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/n3rd0x/Udacity_IntroTensorFlowForDeepLearning/blob/master/010_Saving_and_Loading_Models.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


# Education - Udacity "Intro to TensorFlow for Deep Learning"
# Module: Saving and Loading Models
# REF: https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l07c01_saving_and_loading_models.ipynb


# In[2]:


# Import packages.
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import logging

# Using version 2.x of Tensorflow.
try:
  # Use the %tensorflow_version magic if in colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass

# Tensorflow.
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Set logging level.
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Print Tensorflow version.
print('TensorFlow Version:', tf.__version__)


# In[3]:


# Setup dataset.
print('Download and split the dataset....')
splits = tfds.Split.ALL.subsplit(weighted = (80, 20))
(dataset_training, dataset_validation), dataset_info = tfds.load(
  'cats_vs_dogs',
  with_info = True,
  as_supervised = True,
  split = splits
)
print('Preparing completed.')


# In[4]:


# Display info.
num_classes   = dataset_info.features['label'].num_classes
num_examples  = dataset_info.splits['train'].num_examples

print('-- Total Summary --')
print('Total Classes:    {}'.format(num_classes))
print('Total Exmples:    {}'.format(num_examples))


# In[5]:


# Display some of the samples.
samples = dataset_training.take(5)
plt.figure(figsize = (15, 3))
for i, img in enumerate(samples):
  plt.subplot(1, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img[0])
  plt.xlabel(img[0].shape)
_ = plt.suptitle('Sample of the Training Dataset')


# In[6]:


# MobileNet: 224 x 224 with RGB and normalize [0, 1]
BATCH_SIZE = 32
IMAGE_RES  = 224

print('Resize Process....')

def formatImage(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
  return image, label

batch_training    = dataset_training.shuffle(num_examples//4).map(formatImage).batch(BATCH_SIZE).prefetch(1)
batch_validation  = dataset_validation.map(formatImage).batch(BATCH_SIZE).prefetch(1)
print('Resize Completed.')


# In[7]:


# Build the model.
EPOCHS = 3
URL = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
print('Total Epochs: {}'.format(EPOCHS))
print('--------------------------------------------')
print('-- MobileNet')
print('--------------------------------------------')
print('Create a feature extractor.')
print('URL: {}'.format(URL))

# Create the feature extractor.
# Freeze the extrator so we don't update its pretraing values.
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
feature_extractor.trainable = False
  
# Build our model to classify the flowers.
model = tf.keras.Sequential()
model.add(feature_extractor)
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.summary()


# In[8]:


# Train the model.
# Loss Keyword: sparse_categorical_crossentropy
print('Compile the model.')
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy']
)

print('Start training the model....')
history = model.fit(
  batch_training,
  epochs=EPOCHS,
  validation_data = batch_validation
)
print('--------------------------------------------')


# In[9]:


# Predict testing.
class_names = np.array(dataset_info.features['label'].names)

batch_image, batch_label = next(iter(batch_training.take(1)))
batch_image = batch_image.numpy()
batch_label = batch_label.numpy()

batch_predicted = model.predict(batch_image)
batch_predicted = tf.squeeze(batch_predicted).numpy()

predicted_ids     = np.argmax(batch_predicted, axis=-1)
predicted_classes = class_names[predicted_ids]

print('Predict testing.')
print('Class Names:       {}'.format(class_names))
print('Correct Labels:    {}'.format(batch_label))
print('Predicted IDs:     {}'.format(predicted_ids))
print('Predicted Classes: {}'.format(predicted_classes))


# In[10]:


plt.figure(figsize=(10, 10))
for i in range(30):
  plt.subplot(6, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(batch_image[i])
  color = 'blue' if predicted_ids[i] == batch_label[i] else 'red'
  plt.xlabel('{} ({}%)'.format(predicted_classes[i].title(), round(100 * np.max(batch_predicted[i]))), color = color)
  plt.ylabel(class_names[batch_label[i]].title(), color = 'green')

_ = plt.suptitle("Blue: Correct Prediction  Red: Wrong Prediction  (Green: Correct Label)")


# In[11]:


# Save as Keras model.
keras_file = './pretrained_model_keras.h5'
model.save(keras_file)
print('Save Keras model ({})'.format(keras_file))
get_ipython().system('ls -lh')


# In[12]:


# Load saved Keras model.
loaded_model = tf.keras.models.load_model(
  keras_file,
  # 'custom_object' tells Keras how to load a 'hub.KerasLayer'
  custom_objects = {'KerasLayer' : hub.KerasLayer}
)
loaded_model.summary()


# In[13]:


# Compare the models.
original_batch  = model.predict(batch_image)
loaded_batch    = loaded_model.predict(batch_image)

res = (abs(original_batch - loaded_batch)).max()
print('Comparing Original vs Loaded: {}'.format(res))
print('Here we expect 0.0')
print('Zero means there are no different.')


# In[14]:


# Keep training.
print('Continue traing our loaded model for improving accuracy.')
history = loaded_model.fit(
  batch_training,
  epochs=EPOCHS,
  validation_data = batch_validation
)
print('Training completed.')


# In[15]:


# Compare the models.
original_batch  = model.predict(batch_image)
loaded_batch    = loaded_model.predict(batch_image)

res = (abs(original_batch - loaded_batch)).max()
print('Comparing Original vs Loaded (after training): {}'.format(res))
print('Here we expect non-zero value.')
print('Zero means there are no different.')


# In[16]:


# Export as TensorFlow SavedModel.
path_sm = './tensorflow_saved_model'
get_ipython().system('rm -rf $path_sm/*')
tf.saved_model.save(model, path_sm)
print('Save the model into {}.'.format(path_sm))
get_ipython().system('ls -lh')

print('----------------------')
print('-- {}'.format(path_sm))
print('----------------------')
get_ipython().system('ls -lh {path_sm}')
print('----------------------')


# In[19]:


# Reload TensorFlow SavedModel.
saved_model = tf.saved_model.load(path_sm)

loaded_sm_batch = saved_model(batch_image, training=False).numpy()
res = (abs(original_batch - loaded_sm_batch)).max()
print('Comparing Original vs Loaded SavedModel: {}'.format(res))
print('Here we expect 0.0')
print('Zero means there are no different.')


# In[20]:


# Load SavedModel as Keras model for continue training, if wanted.
loaded_sm_keras = tf.keras.models.load_model(
  path_sm,
  custom_objects = {'KerasLayer': hub.KerasLayer}
)
loaded_sm_keras.summary()


# In[21]:


# Comparing the prediction.
original_batch = model.predict(batch_image)
loaded_sm_batch = loaded_sm_keras.predict(batch_image)
res = (abs(original_batch - loaded_sm_batch)).max()
print('Comparing Original vs Loaded SavedModel (Keras): {}'.format(res))
print('Here we expect 0.0')
print('Zero means there are no different.')


# In[22]:


# Prepare model for download.
print('Preparing TensorFlow SavedModel for download.')
get_ipython().system('zip -r tf_savedmodel.zip {path_sm}')
get_ipython().system('ls -lh')


# In[ ]:


# Download to local disk from Colab.
#try:
#  from google.colab import files
#  files.download('./tf_savedmodel.zip')
#except ImportError:
#  pass

# Or right-click on the respective file
# under the menu 'Files', on the left of Colab.

