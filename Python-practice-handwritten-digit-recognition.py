import keras 
from keras import datasets, models, layers, utils, backend
import cv2, random, os
import numpy as np
from matplotlib import pyplot as plt

backend.set_image_data_format('channels_last')
(X_train, Y_train), (X_value, Y_value) = datasets.mnist.load_data()

X_train = X_train.astype('float32')/255.0
X_value = X_value.astype('float32')/255.0

Y_train = utils.to_categorical(Y_train)
Y_value = utils.to_categorical(Y_value)

number_of_classes = 10 

model = models.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(30, (5,5), strides=(1,1), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax', name= 'predict'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=[X_value, Y_value], epochs=4, batch_size=200)

predition_img = cv2.imread('image.png', 0)

plt.imshow(predition_img, cmap='gray')

if predition_img.shape != (28, 28):
  predition_img = cv2.resize(predition_img, (28, 28))

predition_img = predition_img.reshape(1, 28, 28, 1)
predition_probability = model.predict(predition_img)
predition = np.argmax(predition_probability, axis=-1)
print(f'Predição da Classe: {predition [0]}')
print('predição (probabilities): ')
for probability in predition_probability[0]:
  print(f'{probability:.2f}')
