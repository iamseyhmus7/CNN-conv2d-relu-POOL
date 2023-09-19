import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

import tensorflow_hub as hub
import tensorflow_datasets as tfds
from keras import layers   


(training_set,validation_set), dataset_info = tfds.load(
    'tf_flowers',
    split = ["train[:70%]","train[70%:]"],
    with_info=True,
    as_supervised=True
)

num_classes = dataset_info.features["label"].num_classes # veri kümesinde bulunan etiket sınıflarının sayısını temsil eder. 
num_training_examples = 0 
num_validation_examples = 0

for example in training_set:
    num_training_examples += 1

for example in validation_set:
    num_validation_examples += 1

print('Total Number of Classes: {}'.format(num_classes))
print('Total Number of Training Images: {}'.format(num_training_examples))
print('Total Number of Validation Images: {} \n'.format(num_validation_examples))


# Flowers veri kümesindeki görüntülerin tümü aynı boyutta değildir.

for i , example in enumerate(training_set.take(5)):
      print('Image {} shape: {} label: {}'.format(i+1, example[0].shape, example[1]))

IMAGE_RES = 224 

def format_image(image,label):
     image = tf.image.resize(image,(IMAGE_RES,IMAGE_RES,)) / 255.0
     return image,label


BATCH_SİZE = 32

train_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SİZE).prefetch(1)

validation_batches = validation_set.map(format_image).batch(BATCH_SİZE).prefetch(1)

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(URL,
                                   input_shape = (IMAGE_RES,IMAGE_RES,3))

feature_extractor.trainable = False

model = tf.keras.Sequential([
     feature_extractor,
     layers.Dense(num_classes)
     
])

model.summary()

model.compile(optimizer="adam",
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["Accuracy"])

Epochs = 6

history = model.fit(train_batches,validation_data=validation_batches,epochs=Epochs)


train_acc = history.history["Accuracy"]
val_acc = history.history["val_Accuracy"]

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(Epochs)

plt.figure(figsize=(12,8))
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

class_names = np.array(dataset_info.features["label"].names)
print(class_names)


image_batch , label_batch = next(iter(train_batches))

image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()

predicted_ids = np.argmax(predicted_batch,axis = 1)
predicted_class_names = class_names[predicted_ids]
print(predicted_class_names)

print("Labels: ",label_batch)
print("Predicted labels: ",predicted_ids)



plt.figure(figsize=(12,8))
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(image_batch[n])
    color = "blue" if predicted_ids[n] == label_batch[n] else "red"
    plt.title(predicted_class_names[n].title(),color = color)
    plt.axis("off")
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")   
plt.show()


# YUKARIDAKİ TRANSFER MODELİNİ MOBİLNET İLE GERÇEKLEŞTİRDİK.






# ŞİMDİ İSE İNCEPTİON MODELİ İLE MODEL OLUŞTURUP SONUÇLARI KARŞILAŞTIRALIM


IMAGE_RES = 299

(training_data, validation_data), dataset_info = tfds.load(
    'tf_flowers', 
    with_info=True, 
    as_supervised=True, 
    split=['train[:70%]', 'train[70%:]'],
)

training_batches = training_data.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SİZE).prefetch(1)
validation_batches = validation_data.map(format_image).batch(BATCH_SİZE).prefetch(1)

URL = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,input_shape = (IMAGE_RES,IMAGE_RES,3),
                                   trainable=False)

model_inception = tf.keras.Sequential([
     feature_extractor,
     tf.keras.layers.Dense(num_classes)
])

model_inception.summary()

model_inception.compile(
     optimizer = "adam",
     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
     metrics=["Accuracy"]
)
EPOCHS = 6 
y_pred = model_inception.fit(training_batches,epochs = EPOCHS,validation_data=validation_batches)

train_acc = y_pred.history["Accuracy"]
val_acc = y_pred.history["val_Accuracy"]

train_loss = y_pred.history["loss"]
val_loss = y_pred.history["val_loss"]

epochs_range = range(EPOCHS)

plt.figure(figsize=(12,8))
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



clas_name = np.array(dataset_info.features["label"].names)
print(clas_name)

image_batch , label_batch = next(iter(training_batches))

image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predict_batc = model_inception.predict(image_batch)
predict_batc = tf.squeeze(predict_batc).numpy()

predict_ids = np.argmax(predict_batc,axis = -1)
predict_class_names = clas_name[predict_ids]

print(predict_class_names)

print("Labels:           ", label_batch)
print("Predicted labels: ", predict_ids)



plt.figure(figsize=(12,8))
for i in range(30):
     plt.subplot(6,5,i+1)
     plt.subplots_adjust(hspace=0.3)
     plt.imshow(image_batch[i])
     color = "green" if predict_ids[n] == label_batch[n] else "red"
     plt.title(predict_class_names[n].title(),color = color)
     plt.axis("off")
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
plt.show()





import time

t = time.time()
export_path_keras = "./{}.h5".format(int(t))
print(export_path_keras)
model.save(export_path_keras)


# Şimdi az önce kaydettiğimiz modeli, reloaded adı verilen yeni bir modele yükleyeceğiz. 
# Dosya yolunu ve custom_objects parametresini sağlamamız gerekecek.

reloaded = tf.keras.models.load_model(
     export_path_keras,
     # `custom_objects` keras'a bir `hub.KerasLayer`ın nasıl yükleneceğini anlatır
     custom_objects={"KerasLayer":hub.KerasLayer}
)
reloaded.summary()

#Yeniden yüklenen model ile önceki modelin aynı sonucu verip vermediğini kontrol edebiliriz.

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

(abs(result_batch-reloaded_result_batch)).max()

# Tahmin yapmanın yanı sıra yeniden yüklediğimiz modelimizi alıp eğitmeye de devam edebiliriz.
# Bunu yapmak için, yeniden yüklenenleri her zamanki gibi .fit yöntemini kullanarak eğitebilirsiniz.

EPOCHS = 3
history = reloaded.fit(train_batches,
                       epochs = EPOCHS,
                       validation_data = validation_batches)




t = time.time()
export_path_sm = "./{}".format(int(t))
print(export_path_sm)
tf.saved_model.save(model,export_path_sm)


# LOAD SAVEDMODEL
# Şimdi SavedModel'ımızı yükleyelim ve onu tahminlerde bulunmak için kullanalım.
# SavedModels'ımızı yüklemek için tf.saved_model.load() fonksiyonunu kullanıyoruz.
# tf.saved_model.load tarafından döndürülen nesne, onu oluşturan koddan %100 bağımsızdır.


reloaded_sm = tf.saved_model.load(export_path_sm)
# Şimdi, bir grup görüntü üzerinde tahminlerde bulunmak için reloaded_sm'yi (reloaded SavedModel) kullanalım.
reload_sm_result_batch = reloaded_sm(image_batch,training = False).numpy()
#Yeniden yüklenen SavedModel ile önceki modelin aynı sonucu verip vermediğini kontrol edebiliriz.
(abs(result_batch - reload_sm_result_batch)).max()

