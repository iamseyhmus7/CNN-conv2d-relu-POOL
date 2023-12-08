import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pathlib
import tensorflow_datasets as tfds 
import PIL
import PIL.Image
from keras.preprocessing.image import ImageDataGenerator

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url , extract = True)
data_dir = pathlib.Path(archive).with_suffix('')


image_count = len(list(data_dir.glob("*/*.jpg")))
print(image_count)

class_names = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

roses = list(data_dir.glob("roses/*"))
PIL.Image.open(str(roses[0]))

batch_size = 64
img_width = 180
img_height = 180



datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2, rotation_range=10,
                              width_shift_range=0.1, height_shift_range=0.1,
                             shear_range=0.1, zoom_range=0.10, horizontal_flip=True)


train_ds = datagen.flow_from_directory(data_dir,
                                       subset="training",
                                       target_size = (img_height,img_width),
                                       batch_size=batch_size,
                                       class_mode="categorical")

validation_ds = datagen.flow_from_directory(data_dir,
                                            subset = "validation",
                                            target_size=(img_height,img_width),
                                            batch_size=(batch_size),
                                            class_mode="categorical")


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,Bidirectional

model = Sequential()
model.add(Conv2D(filters=64,kernel_size = (3,3) , input_shape = (180,180,3),activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(64,kernel_size = (3,3),activation=tf.nn.relu))
model.add(MaxPooling2D(3,3))
model.add(Conv2D(64,kernel_size = (3,3),activation="relu"))
model.add(MaxPooling2D(3,3)) 
model.add(Dropout(0.1))
model.add(Flatten())  # Düzleştirme katmanını
model.add(Dense(128,activation=tf.nn.relu))
model.add(Dense(5,activation=tf.nn.softmax))

model.compile(loss = tf.keras.losses.categorical_crossentropy,
              optimizer="adam",metrics=["Accuracy"])


model.summary()



history = model.fit(
  train_ds,
  validation_data=validation_ds,
  epochs=20
)

test_loss, test_accuracy = model.evaluate(validation_ds)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# ACCURACY AND VALİDATİON GRAFİKLEŞTİRELİM.
train_acc = history.history["Accuracy"]
val_acc = history.history["val_Accuracy"]

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(20)  

plt.figure(figsize=(12,10))
plt.subplot(1,2,1)
plt.plot(epochs_range,train_acc,label = "Training Accuracy")
plt.plot(epochs_range,val_acc,label = "Validation Accuracy")
plt.legend(loc = "lower right")
plt.title("Training and Validation Accuracy")


plt.subplot(1,2,2)
plt.plot(epochs_range,train_loss,label = "Training Loss")
plt.plot(epochs_range,val_loss,label = "Validation Loss")
plt.legend(loc = "upper right")
plt.title("Training and Validation Loss")
plt.show()














