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


def main():

    
    # list of the implemented models, second parameter defined is model convolutional or not
    '''

    modelNames = (('mlp_one_layer', 'False'), 
                  ('mlp_two_layers', 'False'), 
                  ('mlp_two_layers_dropout', 'False'),
                  ('conv_net', 'True'),
                  ('conv_net_dropout', 'True'), 
                  ('conv_net_batch_norm', 'True'),
                  ('conv_net_l2', 'True'),
                  ('conv_net_l1', 'True'), 
                  ('leNet', 'True'))
    '''
    
    modelNames = (('conv_net', 'True'),
                  )

    
    # named tuple which is holding hyper parameters values
    HyperParams = namedtuple('HyperParams', 'optimizer epochs batch_size loss is_data_augmented')
    
    hyper_params = HyperParams(optimizer = 'adam', #rmsprop 
                               epochs = 2, 
                               batch_size = 256, 
                               loss = 'categorical_crossentropy',
                               is_data_augmented = False)
             
    reportList = []
    
    print("load dataset ..., the images are encoded as numpy arrays and labels are an array of digits, ranging from 0 to 9")
   
    # load data        
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    for (modelName, isConvModel)  in modelNames:

       # prepare data
        (train_data, train_labels_one_hot), (test_data, test_labels_one_hot) = utils.prepareData(train_images, train_labels, test_images, test_labels, isConvModel, hyper_params.is_data_augmented)
         
        # now data is processed and we are ready to build a network
        model = Model.build(modelName, len(np.unique(test_labels)))
    
        # print model summary
        model.summary()
        
        # compile the model
        model.compile(optimizer = hyper_params.optimizer, loss= hyper_params.loss,  metrics=['accuracy'])
    
        # checkpoint callbacks
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                                    './models/' + utils.getStartTime() + '_' + modelName + '.hdf5',
                                    monitor = 'val_acc',
                                    #save_best_only = True,
                                    verbose = 1,
                                    period = hyper_params.epochs)
        # time callback
        time_callback = TimeCallback()
               
        #print(hyper_params.is_data_augmented)
        #print(isConvModel)
        
        if hyper_params.is_data_augmented == True:# and isConvModel == 'True':
            
            if isConvModel == 'True':

                print("********************************************************************************")
                
                gen = ImageDataGenerator(rotation_range=8, 
                                         width_shift_range=0.08, 
                                         shear_range=0.3,
                                         height_shift_range=0.08, 
                                         zoom_range=0.08)
                
                test_gen = ImageDataGenerator()
                print(train_data.shape)
                train_generator = gen.flow(train_data, train_labels_one_hot, batch_size=64)
                test_generator = test_gen.flow(test_data, test_labels_one_hot, batch_size=64)
                
                history = model.fit_generator(train_generator, 
                                              steps_per_epoch=60000//64, 
                                              epochs = hyper_params.epochs,
                                              verbose = 1, 
                                              callbacks = [time_callback, model_checkpoint_callback], 
                                              validation_data=test_generator, 
                                              validation_steps=10000//64)
                
                
            else:
                x_train = train_data
                y_train = train_labels_one_hot
                input_size = 784
                max_batches = 2 * len(x_train) / hyper_params.batch_size

                gen = ImageDataGenerator(rotation_range=8, 
                                         width_shift_range=0.08, 
                                         shear_range=0.3,
                                         height_shift_range=0.08, 
                                         zoom_range=0.08)
                
                x_train = x_train.reshape(60000, 28, 28, 1)
                #testInputs = testInputs.reshape(10000, 28, 28, 1)

                gen.fit(x_train)
                for e in range(hyper_params.epochs):
                    batches = 0
                    for x_batch, y_batch in gen.flow(x_train, y_train, batch_size = hyper_params.batch_size):
                        
                        x_batch = np.reshape(x_batch, [-1, input_size])
                        
                        #print("X_batch", x_batch.shape)
                        
                        history = model.fit(x_batch, 
                                            y_batch, 
                                            verbose=0,
                                            callbacks = [time_callback, model_checkpoint_callback],
                                            validation_data = (test_data, test_labels_one_hot))
                        
                        batches += 1
                        
                        print(history)
                        print(str(history))
                        
                        print("Epoch %d/%d, Batch %d/%d" % (e+1, hyper_params.epochs, batches, max_batches))
                        
                        if batches >= max_batches:
                            # we need to break the loop by hand because
                            # the generator loops indefinitely
                            break
    
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                #x_test = np.reshape(x_test, [-1, input_size])
                #test_data = x_test
        else:       
            print("------------------------------------------------------------------------------------")
            # fit the model    
            history = model.fit(train_data, 
                                train_labels_one_hot, 
                                batch_size = hyper_params.batch_size, 
                                epochs = hyper_params.epochs, 
                                verbose = 1, 
                                callbacks = [time_callback, model_checkpoint_callback],
                                validation_data = (test_data, test_labels_one_hot))
        
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
               
        #break
                
    utils.generateReport(reportList)
               
        
if __name__ == '__main__':
    main()


