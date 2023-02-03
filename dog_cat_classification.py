import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# %matplotlib inline 
# %config InlineBackend.figure_format = 'retina' #High quality figures


def normalized_rgb(x):
    print(x._count)
    return (x - 128) / 128

### Crear datasets de entrenamiento y pruebas
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory('TrainDogCat/train', 
                                                        labels='inferred',
                                                        validation_split=0.2,
                                                        seed=123,
                                                        subset='training',
                                                        image_size=(img_height,img_width),
                                                        batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory('TrainDogCat/test', 
                                                        labels='inferred',
                                                        validation_split=0.2,
                                                        seed=123,
                                                        subset='validation',
                                                        image_size=(img_height,img_width),
                                                        batch_size=batch_size)

### Almacenar nombres de las etiquetas
class_names = train_ds.class_names


### Mostrar imagenes de entrenamiento
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
#     plt.show()



# normalized_ds = train_ds.take(1).apply(normalized_rgb)



num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
history = model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=epochs
)


### Mostrar rendimiento tras las epocas

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()


### Imagenes a predecir y las pasamos a formato correcto
img_gato = tf.keras.utils.load_img(
    'gato.jpg', target_size=(img_height, img_width)
)

img_perro = tf.keras.utils.load_img(
    'perro.jpg', target_size=(img_height, img_width)
)

img_gato_array = tf.keras.utils.img_to_array(img_gato)
img_gato_array = tf.expand_dims(img_gato_array, 0)

img_perro_array = tf.keras.utils.img_to_array(img_perro)
img_perro_array = tf.expand_dims(img_perro_array, 0)

### Hacemos la prediccion
prediccion = model.predict(img_gato_array)
score = tf.nn.softmax(prediccion[0])
print(score)

print(
    "\n This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)












