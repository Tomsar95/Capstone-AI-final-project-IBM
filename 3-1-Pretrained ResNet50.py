from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
import os


num_classes = 2

image_resize = 224

batch_size_training = 100
batch_size_validation = 100

cur_dir = os.getcwd()

print(cur_dir)

data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_generator = data_generator.flow_from_directory(
    os.path.join(cur_dir, 'concrete_data_week3/train'),
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')

print('flag 1.................................................................')

validation_generator = data_generator.flow_from_directory(
    os.path.join(cur_dir, 'concrete_data_week3/valid'),
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')

model = Sequential()

model.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))

model.add(Dense(num_classes, activation='softmax'))

model.layers

model.layers[0].layers

#Since the ResNet50 model has already been trained, then we want to tell our model not to bother with training the ResNet part, but to train only our dense output layer. To do that, we run the following.
model.layers[0].trainable = False

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)
num_epochs = 2

fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)

model.save('classifier_resnet_model.h5')


print('DONE DONE DONE')