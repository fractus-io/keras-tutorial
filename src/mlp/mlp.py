'''

 Copyright (c) 2018 Fractus IT d.o.o. <http://fractus.io>
 
'''

# importing the libraries  
import numpy as np
import pandas as pd
import keras
import utils
from keras.models import Sequential



    
def main():

    # import dataset 
    dataset = pd.read_csv('Churn_Modelling.csv')

    # prepare data set
    X, Y = utils.prepareData(dataset)

    print(X.shape)
            
    # create model        
    model = utils.getModel()
       
    # compile model   
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # fit the model
    model.fit(X, Y, batch_size = 10, epochs = 100)
    
    model.summary()
    
    # evaluate the model
    scores = model.evaluate(X, Y)

    # print accuracy
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    Y_predict = model.predict(X)
    #Y_predict = (Y_predict > 0.5)
    
    print(type(Y_predict[:10]))

    # print head first 10 rows
    print(Y_predict[:100])

    # print head first 10 rows
    print(Y[:100])

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y, Y_predict)
    print(cm)

if __name__ == '__main__':    
    main()
