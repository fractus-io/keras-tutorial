'''

 Copyright (c) 2020 Fractus IT d.o.o. <http://fractus.io>
 
'''
import keras 
import numpy as np
from keras.applications import vgg16, resnet50, mobilenet, inception_v3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import imagenet_utils

import argparse,textwrap


def main():

    parser = argparse.ArgumentParser()
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Please specify model.',
                                     usage='use "python %(prog)s --help" for more information',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-m', 
                        '--model', 
                        required=True,
                        help= textwrap.dedent('''\
                        Possible MODELS:
                        vgg
                        inception
                        resnet
                        mobilenet                        
                        '''))

    args = vars(parser.parse_args())
    model_name = args["model"]

    if model_name == 'vgg': 
        classify_image_with_vgg()
    elif model_name == 'inception': 
        classify_image_with_inception()
    elif model_name == 'resnet': 
        classify_with_resnet()
    elif model_name == 'mobilenet': 
        classify_image_with_mobilenet()

def classify_image_with_vgg():
    
    model = vgg16.VGG16(weights="imagenet")
    print(model.summary())

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
    predictions = model.predict(processed_image)
    # print predictions
    
    # convert the probabilities to class labels
    # We will get top 5 predictions which is the default
    results = decode_predictions(predictions)
    print(results)

def classify_image_with_inception():

   model = inception_v3.InceptionV3(weights='imagenet')
   print(model.summary())

def classify_with_resnet():

    filename = 'cat.jpg'
    
    image = load_img(filename, target_size=(224, 224))

    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    model = resnet50.ResNet50(weights='imagenet')
    print(model.summary())

    # classify the input image
    predicts = model.predict(image)  
    results = imagenet_utils.decode_predictions(predicts)

    print(results)

def classify_image_with_mobilenet():

    model = mobilenet.MobileNet(weights='imagenet')
    print(model.summary())    

if __name__ == '__main__':
    main()


