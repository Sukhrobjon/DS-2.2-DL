import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image


model = load_model('model.h5')
# The default Graph being used in the current thread.
graph = tf.get_default_graph()

img = Image.open('img_4.jpg')
# convert to array
img = img_to_array(img.resize((48, 48)))
print(f"img shape: {img.shape}")
# reshape args(num of img, size, size, X)
x = img.reshape(1, 3, 48, 48)
x /= 255

with graph.as_default():
    out = model.predict(x)


print(out[0])
r = np.argmax(out[0])
print(f"predicted number: {r}")
