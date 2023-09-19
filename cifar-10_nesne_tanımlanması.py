from keras.models import Sequential
from keras.layers import Dense , MaxPool2D , Conv2D , Flatten
from sklearn.metrics import confusion_matrix , classification_report
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

(X_train , y_train),(X_test , y_test ) = cifar10.load_data()

from keras.utils.np_utils import to_categorical as tc

y_test_tc = tc(y_test , 10)
y_train_tc = tc(y_train , 10)


X_train = X_train.reshape(50000,32,32,3)
X_test = X_test.reshape(10000,32,32,3)

model = Sequential() 
model.add(Conv2D(filters=32 , kernel_size=(3,3),input_shape = (32,32,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D( filters= 64 , kernel_size=(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(64,activation="relu"))
model.add(Dense(10,activation="softmax"))
 

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.summary()

model.fit(X_train,y_train_tc,epochs=10)
eva = model.evaluate(X_test , y_test_tc)
print("Evaluate:",eva)








from PIL import Image
while True:
    img_path = input("Tahmin Etmek istediğiniz Fotoğrafı girin:")
    new_image = Image.open(img_path)
    new_image = new_image.resize((32,32))
    new_image_array = np.array(new_image)
    new_image_array = new_image_array.reshape(1,32,32,3)

    predicted_probs = model.predict(new_image_array)
    prediction_class = np.argmax(predicted_probs,axis = 1)

    print("Prediction Class:",prediction_class)
   
