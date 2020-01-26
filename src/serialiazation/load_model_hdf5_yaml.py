'''

 Copyright (c) 2019 Fractus IT d.o.o. <http://fractus.io>
 
'''

# importing the libraries  
from keras.models import model_from_yaml
import numpy as np

def main():

    # define dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    print("Loaingd model from disk...")

    # load json and create model
    yaml_file = open('model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()

    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # evaluate the model
    scores = model.evaluate(X, Y)

    # print accuracy
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    print("predicting output:")
    print(model.predict_proba(X))

    print("model summary:")
    print(model.summary())

if __name__ == '__main__':    
    main()
