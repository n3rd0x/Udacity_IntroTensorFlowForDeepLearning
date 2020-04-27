#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Education - Udacity "Intro to TensorFlow for Deep Learning"
# Module: Forecasting with Machine Learning
# REF: https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l08c05_forecasting_with_machine_learning.ipynb


# In[2]:


# Import packages.
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

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


# In[ ]:


# ====================
# Helper Function
# ====================
# Plot serie.
def plot_series(time, series, format="-", start=0, end=None, label=None):
  plt.plot(time[start:end], series[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("Value")
  if label:
    plt.legend(fontsize=14)
  plt.grid(True)


# Trend.
def trend(time, slope=0):
  return slope * time


# Seasonal pattern.
def seasonal_pattern(season_time):
  # Just an arbitrary pattern.
  return np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))


# Seasonality.
def seasonality(time, period, amplitude=1, phase=0):
  # Repeats the same pattern at each period.
  season_time = ((time + phase) % period) / period
  return amplitude * seasonal_pattern(season_time)


# White noise.
def white_noise(time, noise_level=1, seed=None):
  rnd = np.random.RandomState(seed)
  return rnd.randn(len(time)) * noise_level


# Dataset for machine learning.
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


# In[4]:


# Create dataset.
time      = np.arange(4 * 365 + 1)
slope     = 0.05
baseline  = 10
amplitude = 40
series    = baseline + trend(time, slope) + seasonality(time, period = 365, amplitude = amplitude)

noise_level = 5
noise       = white_noise(time, noise_level, seed = 42)

series += noise

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()


# In[ ]:


# Prepare for machine learning.
split_time    = 1000
time_train    = time[:split_time]
time_valid    = time[split_time:]
series_train  = series[:split_time]
series_valid  = series[split_time:]


# In[6]:


# ==================
# Linear Model
# ==================
# Clear session, just useful under development, as we usually
# run the session multiple time in the notebook.
EPOCHS      = 100
SEED        = 42
WINDOW_SIZE = 30
tf.keras.backend.clear_session()
tf.random.set_seed(SEED)
np.random.seed(SEED)

dataset_train = window_dataset(series_train, WINDOW_SIZE)
dataset_valid = window_dataset(series_valid, WINDOW_SIZE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1, input_shape = [WINDOW_SIZE])
])
model.compile(
  loss = tf.keras.losses.Huber(),
  optimizer = tf.keras.optimizers.SGD(lr = 1e-5, momentum = 0.9),
  metrics = ["mae"]
)
model.fit(dataset_train, epochs = EPOCHS, validation_data = dataset_valid)


# In[7]:


# Automatically to find an optimal "window size".
# Otherwise we have to manually try different sizes.
tf.keras.backend.clear_session()
tf.random.set_seed(SEED)
np.random.seed(SEED)

dataset_train = window_dataset(series_train, WINDOW_SIZE)
dataset_valid = window_dataset(series_valid, WINDOW_SIZE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1, input_shape = [WINDOW_SIZE])
])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
  lambda epoch: 1e-6 * 10**(epoch / 30)
)
model.compile(
  loss = tf.keras.losses.Huber(),
  optimizer = tf.keras.optimizers.SGD(lr = 1e-6, momentum = 0.9),
  metrics = ["mae"]
)
history = model.fit(dataset_train, epochs = EPOCHS, callbacks = [lr_scheduler])


# In[8]:


# Plot the learning rate.
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-6, 1e-3, 0, 20])


# In[ ]:


# As we see that it would be safe to start with 1e-5.


# In[10]:


# We can auto stop the training when the training starts to
# stop making progress.
tf.keras.backend.clear_session()
tf.random.set_seed(SEED)
np.random.seed(SEED)

dataset_train = window_dataset(series_train, WINDOW_SIZE)
dataset_valid = window_dataset(series_valid, WINDOW_SIZE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1, input_shape = [WINDOW_SIZE])
])

early_stopping = tf.keras.callbacks.EarlyStopping(patience = 10)
model.compile(
  loss = tf.keras.losses.Huber(),
  optimizer = tf.keras.optimizers.SGD(lr = 1e-5, momentum = 0.9),
  metrics = ["mae"]
)
model.fit(
  dataset_train,
  epochs = 500,
  validation_data = dataset_valid,
  callbacks = [early_stopping]
)


# In[ ]:


# Predict function.
def model_forecast(model, series, window_size):
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size, shift=1, drop_remainder=True)
  ds = ds.flat_map(lambda w: w.batch(window_size))
  ds = ds.batch(32).prefetch(1)
  forecast = model.predict(ds)
  return forecast


# In[ ]:


linear_forecast = model_forecast(
  model,
  series[split_time - WINDOW_SIZE : -1],
  WINDOW_SIZE
)[:, 0]


# In[17]:


print("Shape:", linear_forecast.shape)


# In[18]:


plt.figure(figsize = (10, 6))
plot_series(time_valid, series_valid)
plot_series(time_valid, linear_forecast)

mae = tf.keras.metrics.mean_absolute_error(series_valid, linear_forecast).numpy()
print("MAE:", mae)


# In[19]:


# ==================
# Dense Model
# ==================
# Start to find an optimal learning rate.
tf.keras.backend.clear_session()
tf.random.set_seed(SEED)
np.random.seed(SEED)

dataset_train = window_dataset(series_train, WINDOW_SIZE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation = "relu", input_shape = [WINDOW_SIZE]),
  tf.keras.layers.Dense(10, activation = "relu"),
  tf.keras.layers.Dense(1)
])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
  lambda epoch: 1e-7 * 10**(epoch / 20)
)
model.compile(
  loss = tf.keras.losses.Huber(),
  optimizer = tf.keras.optimizers.SGD(lr = 1e-7, momentum = 0.9),
  metrics = ["mae"]
)
history = model.fit(
  dataset_train,
  epochs = EPOCHS,
  callbacks = [lr_scheduler]
)


# In[20]:


plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-7, 5e-3, 0, 30])


# In[ ]:


# Seems good start around 1e-5


# In[21]:


# Start training with early stopping.
tf.keras.backend.clear_session()
tf.random.set_seed(SEED)
np.random.seed(SEED)

dataset_train = window_dataset(series_train, WINDOW_SIZE)
dataset_valid = window_dataset(series_valid, WINDOW_SIZE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation = "relu", input_shape = [WINDOW_SIZE]),
  tf.keras.layers.Dense(10, activation = "relu"),
  tf.keras.layers.Dense(1)
])

early_stopping = tf.keras.callbacks.EarlyStopping(patience = 10)
model.compile(
  loss = tf.keras.losses.Huber(),
  optimizer = tf.keras.optimizers.SGD(lr = 1e-5, momentum = 0.9),
  metrics = ["mae"]
)
model.fit(
  dataset_train,
  epochs = 500,
  validation_data = dataset_valid,
  callbacks = [early_stopping]
)


# In[ ]:


# Forecast.
dense_forecast = model_forecast(
  model, series[split_time - WINDOW_SIZE : -1],
  WINDOW_SIZE
)[:, 0]


# In[23]:


plt.figure(figsize = (10, 6))
plot_series(time_valid, series_valid)
plot_series(time_valid, dense_forecast)

mae = tf.keras.metrics.mean_absolute_error(series_valid, dense_forecast).numpy()
print("MAE:", mae)

