'''

 Copyright (c) 2019 Fractus IT d.o.o. <http://fractus.io>
 
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras import backend as k
import datetime
from bs4 import BeautifulSoup, Tag


_start_time = None
def getStartTime():
    global _start_time
    if _start_time is None:
        _start_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    return _start_time
    
def prepareData(train_images, train_labels, test_images, test_labels, is_conv_model):
        
    trainInputs = train_images
    testInputs = test_images
            
    if is_conv_model == 'True':
        
        # reshape inputs
        # training set has shape (60000, 28, 28), convolutional networks expects shape 28 x 28 x 1
        # new shape will be (60000, 28, 28, 1)

        trainInputs = trainInputs.reshape(60000, 28, 28, 1)
        testInputs = testInputs.reshape(10000, 28, 28, 1)

    elif is_conv_model == 'False':

        # reshape inputs
        # training set has shape (60000, 28, 28), matrix 28x28 will be reshaped to 784 vector
        # new shape will be (60000, 784)
        trainInputs = trainInputs.reshape(60000, 784)
        testInputs = testInputs.reshape(10000, 784)
                
    # covert data to float32
    trainInputs.astype('float32')
    testInputs.astype('float32')
    
    # scale a values of the data from 0 to 1
    trainInputs = trainInputs / 255
    testInputs = testInputs / 255
    
    # Convert the labels from integer to categorical ( one-hot ) vector
    trainLabelsOneHot = to_categorical(train_labels)
    testLabelsOneHot = to_categorical(test_labels)
    
    return (trainInputs, trainLabelsOneHot), (testInputs, testLabelsOneHot)

def drawGraphByType(history, modelName, epochs, type):
    
    startTime = getStartTime()
    
    folder = './outputs/'
    figureName = modelName + '_' + type + '_' + 'e' + str(epochs) + '_' + startTime + '.png'
    
    valType = 'val_' + type
    
    plt.figure()
    plt.style.use("ggplot")
    plt.plot(history.history[type])
    plt.plot(history.history[valType])    
    plt.xlabel('Epoch')
    plt.ylabel(type)
    plt.legend(['Training ' + str(type) + ' : ' +  str(history.history[type][epochs-1]), 'Validation ' +str(type) + ' : ' +  str(history.history[valType][epochs-1])])
    plt.title(str(type) + " (" + str(modelName) + ")")
    #plt.show()
    plt.savefig(folder + figureName, format="png")
    
    return figureName

def drawAccLossGraph(history, modelName, epochs):
            
    startTime = getStartTime()#datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    
    folder = './outputs/'

    figureName = modelName + '_' +  'e_' + str(epochs) + '_' + startTime + '.png'
        
    plt.figure()
    plt.style.use("ggplot")
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['acc'], label='train_acc')
    plt.plot(history.history['val_acc'], label='val_acc')            
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    #plt.legend(['Training ' + str(type) + ' : ' +  str(history.history[type][epochs-1]), 'Validation ' +str(type) + ' : ' +  str(history.history[valType][epochs-1])])
    #plt.title(str(type) + " curves(" + str(modelName) + ")")
    plt.title("Loss and Accuracy ({}".format(str(modelName)))
    #plt.show()
    plt.savefig(folder + figureName, format="png")
    plt.close()
    
    return figureName

def drawTimes(times, modelName, epochs):
            
    startTime = getStartTime()
    
    folder = './outputs/'
    figureName = modelName + '_times_' +  'e_' + str(epochs) + '_' + startTime + '.png'
        
    plt.figure()
    plt.style.use("ggplot")
    plt.plot(times)
    plt.xlabel('Epoch')
    plt.ylabel('Times')

    plt.legend(['Times(seconds): {0:.2f}'.format(sum(times)) ])
    plt.title("Times ({})".format(str(modelName)))

    plt.savefig(folder + figureName, format="png")
    plt.close()
    
    return figureName

def generateReport(reportList):

    startTime =  getStartTime()
    reportName = './outputs/' + startTime + '.html'

    # create bs
    soup = BeautifulSoup('', 'html.parser')
    
    htmlTag = soup.new_tag('html')
    headerTag = soup.new_tag('H1')
    
    headerTag.string = 'Training report at: ' + startTime

    htmlTag.append(headerTag)
        
    for (model, modelName, history, classificationReport, hyper_params, times) in reportList:
    
        epochs = hyper_params.epochs
                    
        drawGraphByType(history, modelName, epochs, 'acc')
        drawGraphByType(history, modelName, epochs, 'loss')
        drawAccLossGraph(history, modelName, epochs)
        drawTimes(times, modelName, epochs)

        reportHeaderTag = soup.new_tag('H2')
        reportHeaderTag.string = 'Model name: ' + modelName
              
        modelSummaryTag = soup.new_tag('P')
        
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
    
        for item in stringlist:
            itemTag = soup.new_tag('P')
            itemTag.string = item
            modelSummaryTag.append(itemTag)
         
        model_plot_name = './outputs/model_plot_' + startTime + '.png'
        #plot_model(model, to_file = './reports/' + model_plot_name, show_shapes=True, show_layer_names=True)

        pTag = soup.new_tag('p')
        imageTag = soup.new_tag('img')
        imageTag['src'] = model_plot_name
        pTag.append(imageTag)
        modelSummaryTag.append(pTag)
            

        # print thet classification report
        stringlist = classificationReport.split('\n')    
        classificationReportTag = soup.new_tag('table')
        for row in stringlist:
            trTag = soup.new_tag('tr')
            tdTag = soup.new_tag('td')
            tdTag.string = row
            trTag.append(tdTag)
            classificationReportTag.append(trTag)
        
        htmlTag.append(reportHeaderTag)
        htmlTag.append(modelSummaryTag)
        htmlTag.append(classificationReportTag)

    soup.append(htmlTag)
        
    # save bs as html file
    htmlFile = soup.prettify("utf-8")
    with open(reportName, "wb") as file:
        file.write(htmlFile)
    

    
        
