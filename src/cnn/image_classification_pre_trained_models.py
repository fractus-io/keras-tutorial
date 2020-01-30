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
from keras.applications.mobilenet import preprocess_input 
from keras.applications.inception_v3 import preprocess_input
import argparse,textwrap

'''
script shows how to use pretrained Keras models in order to classify image
'''
def main():

    parser = argparse.ArgumentParser()

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

    image = load_image()
    image = vgg16.preprocess_input(image)
    
    predictions = model.predict(image)
    print(decode_predictions(predictions))

def classify_image_with_inception():

    model = inception_v3.InceptionV3(weights='imagenet')
    print(model.summary())

    image = load_image()
    image = inception_v3.preprocess_input(image)

    predicts = model.predict(image)  
    print(imagenet_utils.decode_predictions(predicts))


def classify_with_resnet():

    model = resnet50.ResNet50(weights='imagenet')
    print(model.summary())

    image = load_image()
    image = imagenet_utils.preprocess_input(image)

    predicts = model.predict(image)  
    print(imagenet_utils.decode_predictions(predicts))

def classify_image_with_mobilenet():

    model = mobilenet.MobileNet(weights='imagenet')
    print(model.summary())

    image = load_image()
    image = mobilenet.preprocess_input(image)

    predicts = model.predict(image)  
    print(imagenet_utils.decode_predictions(predicts))

def load_image():

    filename = 'cat.jpg'

    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

if __name__ == '__main__':
    main()


