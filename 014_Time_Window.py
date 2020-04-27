#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Education - Udacity "Intro to TensorFlow for Deep Learning"
# Module: Time Window
# REF: https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l08c04_time_windows.ipynb


# In[2]:


# Import packages.
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

# Using version 2.x of Tensorflow.
try:
  # Use the %tensorflow_version magic if in colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass

# Tensorflow.
import tensorflow as tf

# Print Tensorflow version.
print('TensorFlow Version:', tf.__version__)


# In[3]:


# Create a generated data range.
# Values [0, 10]
dataset = tf.data.Dataset.range(10)

print("--------------------")
print("- Dataset")
print("--------------------")
for v in dataset:
  print(v.numpy())
print("--------------------")


# In[4]:


# Create a window of the dataset.
# Split into 5 values, with shifting of 1.
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1)

print("--------------------")
print("- Window of Dataset")
print("--------------------")
for w in dataset:
  for v in w:
    # Specify end to make it prints on the same line.
    print(v.numpy(), end=" ")
  print()
print("--------------------")


# In[5]:


# Create a window with same fixed size.
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)

print("--------------------")
print("- Fixed Window Size")
print("--------------------")
for w in dataset:
  for v in w:
    # Specify end to make it prints on the same line.
    print(v.numpy(), end=" ")
  print()
print("--------------------")


# In[6]:


# Each batch of the window is a dataset, but we rather want
# batches in form of regular tensor.
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)

# The dataset contains a tensor with size of 5.
dataset = dataset.flat_map(lambda window: window.batch(5))

print("--------------------")
print("- Batch of Window")
print("--------------------")
for w in dataset:
  print(w.numpy())
print("--------------------")


# In[7]:


# For machine learning we want input data with respective label.
# Here we use the first 4 values as inputs, and the last one as labels.
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)

# The dataset contains a tensor with size of 5.
dataset = dataset.flat_map(lambda window: window.batch(5))

# The dataset contains arrays of inputs and repective labels.
dataset = dataset.map(lambda window: (window[ : -1], window[-1 : ]))

print("--------------------")
print("- Arrays of Dataset")
print("--------------------")
for x, y in dataset:
  print(x.numpy(), y.numpy())
print("--------------------")


# In[8]:


# We also want the input to be shuffled and identical distributed.
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)

# The dataset contains a tensor with size of 5.
dataset = dataset.flat_map(lambda window: window.batch(5))

# The dataset contains arrays of inputs and repective labels.
dataset = dataset.map(lambda window: (window[ : -1], window[-1 : ]))

# Shuffle the dataset.
dataset = dataset.shuffle(buffer_size = 10)

print("--------------------")
print("- Shuffled Dataset")
print("--------------------")
for x, y in dataset:
  print(x.numpy(), y.numpy())
print("--------------------")


# In[9]:


# We want to split the dataset into batches, usually 32.
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)

# The dataset contains a tensor with size of 5.
dataset = dataset.flat_map(lambda window: window.batch(5))

# The dataset contains arrays of inputs and repective labels.
dataset = dataset.map(lambda window: (window[ : -1], window[-1 : ]))

# Shuffle the dataset.
dataset = dataset.shuffle(buffer_size = 10)

# Create batches. The 'prefetch' indicates the Tensorflow will
# fetch the data while working, so we always have data to use.
dataset = dataset.batch(2).prefetch(1)

print("--------------------")
print("- Batch of Dataset")
print("--------------------")
for x, y in dataset:
  print("x:", x.numpy())
  print("y:", y.numpy())
print("--------------------")


# In[10]:


# Create a function for these steps.
# This converts a time series into a dataset for machine learning.
def window_dataset(series, window_size, batch_size = 32, shuffle_buffer = 1000):
  # Create a dataset of tensors.
  dataset = tf.data.Dataset.from_tensor_slices(series)

  # Drop remaining values so we get same size.
  dataset = dataset.window(window_size + 1, shift = 1, drop_remainder = True)

  # The dataset contains a tensor with size of 5.
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

  # Split into two arrays, one for inputs and one for labels.
  dataset = dataset.map(lambda window: (window[ : -1], window[-1 : ]))

  # Shuffle the dataset.
  dataset = dataset.shuffle(shuffle_buffer)

  # Create batches. The 'prefetch' indicates the Tensorflow will
  # fetch the data while working, so we always have data to use.
  dataset = dataset.batch(batch_size).prefetch(1)

  return dataset

dataset = window_dataset(np.arange(10), 4, 2, 10)
print("--------------------")
print("- Dataset")
print("--------------------")
for x, y in dataset:
  print("x:", x.numpy())
  print("y:", y.numpy())
print("--------------------")

