import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('handwritten.model')

try:
    
    img = cv2.imread("./demo.png")[:,:,0]
    img = np.array([img])
    prediction = model.predict(img)
    print(prediction)
    print(f"This digit is probably a {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
except:
    print('Error. Please check the image resolution.')