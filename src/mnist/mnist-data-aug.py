'''

 Copyright (c) 2019 Fractus IT d.o.o. <http://fractus.io>
 
'''
import utils
import numpy as np
import matplotlib.pyplot as plt
import keras

from collections import namedtuple

from model import Model
from callbacks import TimeCallback

from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as k
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
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
                        Possible MODEL:
                        conv_net
                        conv_net_dropout
                        conv_net_batch_norm
                        conv_net_l1
                        conv_net_l2
                        all -> will train all models listed above
                        '''))

    args = vars(parser.parse_args())
    model_name = args["model"]

    # list of the implemented models, second parameter defined is model convolutional or not
    existing_model_names = (('conv_net', 'True'),
                            ('conv_net_dropout', 'True'), 
                            ('conv_net_batch_norm', 'True'),
                            ('conv_net_l1', 'True'),
                            ('conv_net_l2', 'True'))

    if (any(model_name in i for i in existing_model_names)) : 
        # if model name is supported
        tmp_dict = dict(existing_model_names)
        modelNames = ((model_name, tmp_dict[model_name]),)
    elif (model_name == 'all'):
        modelNames = existing_model_names
    else : 
        # if model name is not supported
        print("MODEL:" + model_name + ' is not supported, please speficy correct MODEL') 
        exit()

    
    # named tuple which is holding hyper parameters values
    HyperParams = namedtuple('HyperParams', 'optimizer epochs batch_size loss')
    
    hyper_params = HyperParams(optimizer = 'adam', #rmsprop 
                               epochs = 2, 
                               batch_size = 256, 
                               loss = 'categorical_crossentropy')
             
    reportList = []
    
    print("load dataset ..., the images are encoded as numpy arrays and labels are an array of digits, ranging from 0 to 9")
   
    # load data        
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    for (modelName, isConvModel)  in modelNames:

       # prepare data
        (train_data, train_labels_one_hot), (test_data, test_labels_one_hot) = utils.prepareData(train_images, train_labels, test_images, test_labels, isConvModel)
         
        # now data is processed and we are ready to build a network
        model = Model.getModel(modelName)
    
        # print model summary
        model.summary()
        
        # compile the model
        model.compile(optimizer = hyper_params.optimizer, loss= hyper_params.loss,  metrics=['accuracy'])
    
        # checkpoint callbacks
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                                    './outputs/' + utils.getStartTime() + '_' + modelName + '.h5',
                                    monitor = 'val_acc',
                                    verbose = 1,
                                    period = hyper_params.epochs)

        # time callback
        time_callback = TimeCallback()
               
        gen = ImageDataGenerator(rotation_range=8, 
                                    width_shift_range=0.08, 
                                    shear_range=0.3,
                                    height_shift_range=0.08, 
                                    zoom_range=0.08)
        
        test_gen = ImageDataGenerator()

        train_generator = gen.flow(train_data, train_labels_one_hot, batch_size=64)
        test_generator = test_gen.flow(test_data, test_labels_one_hot, batch_size=64)
        
        history = model.fit_generator(train_generator, 
                                        steps_per_epoch=60000//1000, 
                                        epochs = hyper_params.epochs,
                                        verbose = 1, 
                                        callbacks = [time_callback, model_checkpoint_callback], 
                                        validation_data=test_generator, 
                                        validation_steps=10000//1000)


        [test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
        
        print("Baseline Error: {} test_loss {} , test_acc {}  ".format((100-test_acc*100), test_loss, test_acc))    
                                
        prediction = model.predict(test_data, hyper_params.batch_size)
        
        le = LabelBinarizer()
        le_train_labels = le.fit_transform(train_labels)
        le_test_labels = le.transform(test_labels)

        classificationReport = classification_report(
                                    le_test_labels.argmax(axis=1),
                                    prediction.argmax(axis=1),
                                    target_names=[str(x) for x in le.classes_])
        
        print(classificationReport)
        
        reportList.append([model, modelName, history, classificationReport, hyper_params, time_callback.times])
                              
    utils.generateReport(reportList)
               
        
if __name__ == '__main__':
    main()


