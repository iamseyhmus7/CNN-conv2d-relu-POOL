import tensorflow as tf
import tensorflow_datasets as tfds
import os 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#cats_vs_dogs alı veri kümesini yükledik
dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)
class_names = info.features['label'].names
print(class_names)


for i, example in enumerate(dataset['train']):
  # example = (image, label)
  image, label = example
  save_dir = './cats_vs_dogs/train/{}'.format(class_names[label])
  os.makedirs(save_dir, exist_ok=True)

  filename = save_dir + "/" + "{}_{}.jpg".format(class_names[label], i)
  tf.keras.preprocessing.image.save_img(filename, image.numpy())
  # print(filename)
  # break



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential  


datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2, rotation_range=10,
                              width_shift_range=0.1, height_shift_range=0.1,
                             shear_range=0.1, zoom_range=0.10, horizontal_flip=True)

train_generator = datagen.flow_from_directory('./cats_vs_dogs/train',
                                              target_size = (150, 150), 
                                              batch_size=128, 
                                              class_mode='binary',
                                              subset='training')

validation_generator = datagen.flow_from_directory('./cats_vs_dogs/train',
                                              target_size = (150, 150), 
                                              batch_size=128, 
                                              class_mode='binary',
                                              subset='validation')



# build CNN Model

from keras.backend import batch_normalization

model = Sequential()

# 1st Layer CNN
model.add(Conv2D(32,kernel_size = 4 , activation = "relu",input_shape =(150,150,3)))
model.add(MaxPooling2D(3))
model.add(BatchNormalization())
model.add(Dropout(0.4))

#2nd LAYER CNN
model.add(Conv2D(64,kernel_size = 4,activation = "relu"))
model.add(MaxPooling2D(3))
model.add(BatchNormalization())
model.add(Dropout(0.4))

#3rd LAYER CNN
model.add(Conv2D(128,kernel_size = 4,activation = "relu"))
model.add(MaxPooling2D(3))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256,activation = "relu"))
model.add(Dense(1,activation = "sigmoid"))

model.summary()
model.compile(loss="binary_crossentropy",optimizer = "adam",metrics = ["accuracy"])
history = model.fit(train_generator,epochs = 10 , validation_data = validation_generator)



history.history

plt.plot(history.history["accuracy"],label = "Training")
plt.plot(history.history["val_accuracy"],label = "Validation")
plt.legend(["Training","Validation"])

# #SAVE MODEL
# model.save("cats_vs_dog.h5")

# # Eğitilmiş modeli yükle
# model_load = tf.keras.models.load_model("cats_vs_dog.h5")


import requests
from PIL import Image
from tensorflow.keras.preprocessing import image


def cats_vs_dogs():
    img_url = "https://i.natgeofe.com/n/548467d8-c5f1-4551-9f58-6817a8d2c45e/NationalGeographic_2572187_square.jpg"
    print("Sınıflandırma yapılacak resim: {}".format(img_url))

    # Resmi indirip yeniden boyutlandırma işlemi
    img = Image.open(requests.get(img_url, stream=True).raw).resize((150, 150))

    # Resmi modele uygun hale getirme
    image_array = np.array(img)
    img = np.expand_dims(image_array, axis=0)
    img = img / 255

    # Tahmin yapma
    prediction = model.predict(img)
    TH = 0.5
    prediction = int(prediction[0][0] > TH)

    # Sınıf adları
    classes = {v: k for k, v in train_generator.class_indices.items()}

    # Tahmin sonucunu yazdırma
    predicted_classes = classes[prediction]
    print("Tahmin Edilen Sınıf:", predicted_classes)

# Fonksiyonu çağırmayı unutmayın
cats_vs_dogs()












