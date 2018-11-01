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

    # prepareDate
    X, Y = utils.prepareData(dataset)
            
    # create model        
    model = utils.getModel()
       
    # compile model   
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # fir the model
    model.fit(X, Y,  validation_split=0.33, batch_size = 10, epochs = 100)
    
    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
if __name__ == '__main__':    
    main()

