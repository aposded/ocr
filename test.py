import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import train

model = tf.keras.models.load_model('handwritten.model')
(x_test, y_test) = (train.x_test, train.y_test)
loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)
