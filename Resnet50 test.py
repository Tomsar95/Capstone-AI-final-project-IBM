from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Dense
# from keras.applications import ResNet50
# from keras.applications.resnet50 import preprocess_input
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
import os
from keras.models import load_model

# In this part, you will evaluate your deep learning models on a test data. For this part, you will need to do the following:

# Load your saved model that was built using the ResNet50 model.
# Construct an ImageDataGenerator for the test set. For this ImageDataGenerator instance, you only need to pass the directory of the test images, target size, and the shuffle parameter and set it to False.
# Use the evaluate_generator method to evaluate your models on the test data, by passing the above ImageDataGenerator as an argument. You can learn more about evaluate_generator here.
# Print the performance of the classifier using the VGG16 pre-trained model.
# Print the performance of the classifier using the ResNet pre-trained model.

num_classes = 2

image_resize = 224

# batch_size_training = 100
# batch_size_validation = 100

cur_dir = os.getcwd()

print(cur_dir)


modelresnet50 = load_model(os.path.join(cur_dir, 'classifier_resnet_model.h5'))
#modelVGG16 = load_model(os.path.join(cur_dir, 'classifier_VGG16_model.h5'))

data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

test_generator = data_generator.flow_from_directory(
    os.path.join(cur_dir, 'concrete_data_week4/test'),
    target_size=(image_resize, image_resize),
    class_mode='categorical',
    shuffle=False)

steps_per_epoch_test = len(test_generator)

print('len test generator = '+str(steps_per_epoch_test))

eval_gen = modelresnet50.evaluate_generator(test_generator, steps = steps_per_epoch_test ,verbose=1)

print('FLAG 1 ---------------------------------')

print(eval_gen)

print("Accuracy = ",eval_gen[1])

#fit_history_VGG16 = modelVGG16.fit_generator(
#    test_generator,
#    verbose=1,
#)

# fit_history_resnet50 = modelresnet50.fit_generator(
#     test_generator,
#     verbose=1,
# )

#print(fit_history_VGG16)
# print(fit_history_resnet50)

predict = modelresnet50.predict_generator(test_generator, steps=steps_per_epoch_test, verbose=1)

print('FLAG 2 ---------------------------------')

print(predict)





print('DONE DONE DONE')