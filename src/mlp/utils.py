'''

 Copyright (c) 2018 Fractus IT d.o.o. <http://fractus.io>
 
'''
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

def getModel():

    model = Sequential()
    
    # Adding the input layer and the first hidden layer
    model.add(Dense(6, activation = 'relu', input_dim = 11))
    # Adding the second hidden layer
    model.add(Dense(6, activation = 'relu'))
    # Adding the output layer
    model.add(Dense(1, activation = 'sigmoid'))

    return model

def prepareData(dataset):

   
    # dataset has 14 columns as follows:
    # nmr name
    # 1:  RowNumber    
    # 2:  CustomerId    
    # 3:  Surname    
    # 4:  CreditScore    
    # 5:  Geography    
    # 6:  Gender    
    # 7:  Age    
    # 8:  Tenure    
    # 9:  Balance    
    # 10: NumOfProducts    
    # 11: HasCrCard    
    # 12: IsActiveMember    
    # 13: EstimatedSalary    
    # 14: Exited

    # Create matrix of features and matrix of target variable. 
    # Column RowNumber and CustomerId are not useful in our analysis, so we will exclude them
    # Column Exited is our Target Variable(Y)    
    X = dataset.iloc[:, 3:13].values
    Y = dataset.iloc[:, 13].values
    
    
    # Column Country has string labels such as France, Spain while column Gender has values Male, Female. 
    # We have to encode those strings into numeric. 
    # We will use LabelEncoder from Scikitlearn
    # Labels are encoded in values between 0 to n_classes-1    
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
    

    # Country column needs additional encoding, so we will use OneHotEncoder from ScikitLearn   
    onehotencoder = OneHotEncoder(categorical_features = [1])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]
            
    # Feature Scaling
    #  Data are scaled properly. 
    # Some variable has value in thousands while some have value is tens or ones. 
    # We will use StandarsScvaler from ScikitLearn.
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    return X, Y
