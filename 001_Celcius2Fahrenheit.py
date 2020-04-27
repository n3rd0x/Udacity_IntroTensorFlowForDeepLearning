#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import packages.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set log error verbosity.
tf.logging.set_verbosity(tf.logging.ERROR)


# In[8]:


# Setup the training data.
data_celcius_in = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
data_fahrenheit_out = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Print the data.
print("Machine Learning termonology:")
print("Inputs (Celcius) are called 'Feature'")
print("Outputs (Fahrenheit) are called 'Labels'")
print("")
print("-- Data List ---------------------------")
for i,c in enumerate(data_celcius_in):
  print("{} degrees Celcius = {} degrees Fahrenheit".format(c, data_fahrenheit_out[i]))
print("----------------------------------------")


# In[ ]:


# Create the model.
layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])

# Compile the model.
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))


# In[12]:


# Train the model.
print("Start training the model.")
print("The training would be 3500 iterations (7 'inputs' x 500 'epochs').")
history = model.fit(data_celcius_in, data_fahrenheit_out, epochs=500, verbose=False)
print("Finished traning the model.")


# In[14]:


# Plot the training statistics.
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])


# In[25]:


# Use the model to predict the output.
print("The formula: F = C * 1.8 + 32")
answer = (100 * 1.8) + 32
print("100.0 C = {} (Answer: {})".format(model.predict([100.0]), answer))


# In[27]:


# Print internal layer data.
print("-- Layer Data --------------------------")
print(layer.get_weights())
print("----------------------------------------")


# In[32]:


# Creating a new model with three layers.
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
print("Start training the model.")
model.fit(data_celcius_in, data_fahrenheit_out, epochs=500, verbose=False)
print("Finished training the model.")
print("100.0 C = {} (Answer: {})".format(model.predict([100.0]), answer))
print("-- Layer Data (0) ----------------------")
print(l0.get_weights())
print("----------------------------------------")
print("-- Layer Data (1) ----------------------")
print(l1.get_weights())
print("----------------------------------------")
print("-- Layer Data (2) ----------------------")
print(l2.get_weights())
print("----------------------------------------")

