import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDat Generator
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('content//sign_mnist_train.csv')
test = pd.read_csv('content//sign_mnist_test.csv')

#display(train.info())
#display(test.info())

train_Y = train['label']
train_X = train.drop(['label'], axis=1)

test_Y = test['label']
test_X = test.drop(['label'], axis =1)

train_X = train_X.values.reshape((train_X.shape[0], 28, 28, 1)).astype('float32') / 255
test_X = test_X.values.reshape((test_X.shape[0], 28, 28, 1)).astype('float32') / 255
train_Y = to_categorical(train_Y, num_classes=26)
test_Y = to_categorical(test_Y, num_classes=26)

datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

datagen.fit(train_X)

model = Sequential()

model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(26, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    datagen.flow(train_X, train_Y, batch_size=64),
    epochs=8,
    validation_data=(test_X, test_Y)
)

test_loss, test_accuracy = model.evaluate(test_X, test_Y)
print('Test accuracy:', test_accuracy)

img = test_X[1]

test_img = img.reshape((1, 28, 28, 1))
img_class = model.predict(test_img)
prediction_index = np.argmax(img_class, axis=-1)[0]

actual_label = np.argmax(test_Y[prediction_index])
print("Class:", actual_label)

plt.imshow(img)
plt.title(actual_label)
plt.show()

model.save('asl-cnn-v2.keras')
#from google.colab import files
#files.download('asl-cnn-v2.keras')
  