'''

 Copyright (c) 2020 Fractus IT d.o.o. <http://fractus.io>
 
'''
import keras 
import numpy as np
from keras.applications import vgg16, resnet50, mobilenet, inception_v3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
 
def main():

    print("hello")

    vgg_model = vgg16.VGG16(weights="imagenet")
    
    inception_model = inception_v3.InceptionV3(weights='imagenet')
    
    resnet_model = resnet50.ResNet50(weights='imagenet')
    
    mobilenet_model = mobilenet.MobileNet(weights='imagenet')

    filename = 'cat.jpg'
    
    original = load_img(filename, target_size=(224, 224))
    
    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)
    
    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)

    # prepare the image for the VGG model
    processed_image = vgg16.preprocess_input(image_batch.copy())
    
    # get the predicted probabilities for each class
    predictions = vgg_model.predict(processed_image)
    # print predictions
    
    # convert the probabilities to class labels
    # We will get top 5 predictions which is the default
    label = decode_predictions(predictions)
    print(label)
        
if __name__ == '__main__':
    main()


