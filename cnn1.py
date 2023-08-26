import pandas as pd
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt



#parametreyi ayarla
plt.rc("figure",autolayout = True)
plt.rc("image",cmap = "magma")


# çekirdeği tanımlayın
kernel = tf.constant([[-1,-1,-1],
                      [-1,8,-1],
                      [-1,-1,-1]])


# resmi yükle
image = tf.io.read_file("Ganesh.jpg ")
image = tf.io.decode_jpeg(image , channels=1)
image = tf.image.resize(image,size = [1000,800])

# resmin grafiğini çizin
img = tf.squeeze(image).numpy()
plt.figure(figsize = (12,8))
plt.imshow(img,cmap = "gray")
plt.axis("off")
plt.title("Original Gray Scale Resim")
plt.show()

# Reformat
image = tf.image.convert_image_dtype(image,dtype = tf.float32)
image = tf.expand_dims(image,axis = 0)
kernel = tf.reshape(kernel,[*kernel.shape,1,1])
kernel = tf.cast(kernel,dtype=tf.float32)

# evrişim katmanı
conv_fn = tf.nn.conv2d

image_filter = conv_fn(
    input = image ,
    filters=kernel,
    strides = 1,
    padding = "SAME"
)


plt.figure(figsize = (12,8))

# Evrişimli görüntünün grafiğini çizin
plt.subplot(1,3,1)

plt.imshow(
    tf.squeeze(image_filter)
)
plt.axis("off")
plt.title("Convolution")

# activation layer
relu_fn = tf.nn.relu

# Image detection
image_detect = relu_fn(image_filter)

plt.subplot(1,3,2)
plt.imshow(
    # Çizim için yeniden biçimlendirme
    tf.squeeze(image_detect)
)

plt.axis("off")
plt.title("Activation")

# Pooling layer
pool = tf.nn.pool

image_condense = pool(input = image_detect,
                      window_shape=(2,2),
                      pooling_type="MAX",
                      strides = (2,2),
                      padding = "SAME")

plt.subplot(1,3,3)
plt.imshow(tf.squeeze(image_condense))
plt.axis("off")
plt.title("Pooling")
plt.show()


