
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



# Define data augmentation options
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255  # Normalization
)

# Flowing data from directory
train_generator = datagen.flow_from_directory(
    'asl-dataset\\train',
    target_size=(150, 150),
    color_mode = "grayscale",
    batch_size=32,
    class_mode='sparse'
)



test_generator = datagen.flow_from_directory(
    'asl-dataset\\val',
    target_size=(150, 150),
    color_mode = "grayscale",
    batch_size=32,
    class_mode='sparse'
)


# Define your model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(24, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using the training generator
model.fit(train_generator, epochs=4, validation_data=test_generator)

# Evaluate the model using the testing generator
models.save_model(model,"gray_model.keras")
test_loss, test_accuracy = model.evaluate(test_generator)
print('Test accuracy:', test_accuracy)
