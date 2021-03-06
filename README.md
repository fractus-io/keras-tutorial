

# Keras Tutorial

#### Table of Content

#### Introduction
* [Introduction To Deep Learning](#1)
* Biologicals
* Artificial Neural Networks - Forward Propagation

#### Deep Learning with Keras
* [Introduction to Keras](#31)
  * [Keras, TensorFlow, Theano, and CNTK](#311)
  * [Installing Keras](#312)
  	* [Install useful dependencies](#3121)
	* [Install Theano](#3122)
	* [Install Tensorflow](#3123)
	* [Install Keras](#3124)
  * [Configuring Keras](#313)
* [Develop Your First Neural Network with Keras, XOR gate](#32)
  * [Experiment](#321)
* [Customer Churn Prediction with Multi Layer Perceptrons](#33)
  * [Load data](#331)
  * [Prepare data](#332)
  * [Define Model](#333)
  * [Compile Model](#334)
  * [Fit Model](#335)
  * [Evaluate Model](#336)
  * [Experiment](#337)
* [Summary](#34)


#### Deep Learning Models
* [Keras Models](#41)
    * [Sequential API](#412)
    * [Layers](#413)
    * [Activation functions](#414)
    * [How to Choose Activation Function](#415)
    * [Initializers](#416)
    * [Compilations](#417)
    * [Optimizers](#418)
    * [Loss functions](#419)
    * [Training](#4110)
    * [Evaluate](#4111)
    * [Predict](#4112)
 * [Summary](#42)

#### Model Optimization
* [MNIST Handwritten Digits classifier](#511)
  * [A quick note on **image_data_format**](#5111)
  * [Load MNIST Dataset](#5112)
* [Define & train Model, Multy Layer Perceptron(one layer)](#512)   
  * [Callbacks](#5121)
  * [Evaluate trained model](#5122)
  * [Understand Model Behavior During Training By Plotting History](#5123)
* [Multy Layer Perceptron with two hidden layers](#513)
* [Convolutional Neural Network](#514)
* [Techniques to reduce Overfitting](#515)
  * [L2 & L1 regularization](#5151)
  * [Batch Normalization](#5152)
  * [Dropouts](#5153)
  * [Data Augmentation](#5154)
* [Experiment](#516)  
* [Summary](#517)

#### Model Serialization
* [Introduction](#61)
* [Save model](#62)
* [Load model](#63)
* [Expose model via REST API](#64)
* [Experimet](#65)
* [Summary](#66)

#### Visualizing Neural Networks
* [Introduction](#71)
* [Visualizing Neural Networks with Keras](#72)
* [Summary](#73)

#### Convolutional Neural Networks

* [Introduction](#81)
* [What are Convolutional Neural Networks and what motives their use?](#82)
* [The History of Convolutional Neural Networks](#83)
* [Building blocks of Convolutional Neural Networks](#84)
  * [Convolutional layers](#841)
  * [Pooling layers](#842)
  * [Fully connected layers](#843)
* [Summary](#85)
* [LeNet MNIST](#86)
* [Transfer learning](#87)

#### Recurrent Neural Networks
* [Introduction](#91)
* [The Sequencing Problem](#92)
* [Recurrent Neural Networks](#93)
* [A Special Kind of RNNs: Long Short-Term Memory Networks](#94)
* [Applications of Recurrent Neural Networks](#95)
  * [Energy Forecasting using Time-series prediction](#951)
  * [Speech Recognition using Recurrent Neural Networks](#952)
  * [Recurrent Neural Networks for the purpose of Natural Language Processing (NLP)](#953)
* [Summary](#96)

....

* Shallow and Deep Neural Networks
* Model Architectures
* Train the Model
* Serialize the Model
* undefitting ?

#### Model Optimization
* Understand Model Behavior During Training
* Reduce Overfitting with Dropout Regularization
* Lift Performance with Learning Rate Schedules

#### Applied Deep Learning
* Convolutional Neural Networks
* Recurrent Neural Networks
* Time Series Prediction

#### <a id="1"></a>Introduction To Deep Learning
...

# Deep Learning with Keras
## <a id="31"></a>Introduction To Keras

Keras is a high-level neural networks API, written in Python and which can be run on top of the TensorFlow, CNTK, or Theano. 
It was developed with a focus on enabling fast experimentation. 
Keras runs on Python 2.7 or 3.6 and can easily be executed on GPUs and CPUs. 
Keras was initially developed for researchers, with the aim of enabling fast experimentation.

Keras has following features:

* easy and fast prototyping (through user friendliness, modularity, and extensibility)
* supports both convolutional networks and recurrent networks, as well as combinations of the two
* runs on CPU and GPU.

Keras was developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System), 
and its primary author and maintainer is François Chollet, a Google engineer.

According to Keras documentation, the framework is based on following guiding principles:
* **User friendliness**    
	Keras is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple 
	APIs, it minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.

* **Modularity**    
	A model is understood as a sequence or a graph of standalone, fully-configurable modules that can be plugged together with as few restrictions 
	as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions, 
	regularization schemes are all standalone modules that you can combine to create new models.

* **Extensibility**   
	New modules are easy to add (as new classes and functions). 

* **Python**  
	Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.

Keras is distributed under MIT license, meaning that it can be freely used in commercial projects. It's compatable with any version of the 
Python from 2.7 to 3.6.
Keras has more then 200 000 users, and it is used by Google, Netflix, Uber and hundreds if startups. Keras is also popular framework on Kaggle, 
the machine-learning competition website, where almost every deep learning competition has been won using Keras models.

### <a id="311"></a>Keras, TensorFlow, Theano, and CNTK

Keras provides high-level building blocks for developing deep-learning models. Instead of handling low-level operations such as tensor manipulation, 
Keras relies on a specialized, well-optimized tensor library to do so, serving as the back-end engine of Keras.

Several different back-end engines can be plugged into Keras. 
Currently, the three existing back-end implementations are:
* TensorFlow back-end
* Theano back-end
* Microsoft Cognitive Toolkit (CNTK) back-end. 

Those back-ends are some of the primary platforms for deep learning these days. 

Theano (http://deeplearning.net/software/theano) is developed by the MILA lab at Université de Montréal, 
TensorFlow (www.tensorflow.org) is developed by Google,
and CNTK (https://github.com/Microsoft/CNTK) is developed by Microsoft. 

Any piece of code is developed with Keras can be run with any of these back-ends without having to change anything in the code.

During our tutorial we will use TensorFlow back-end for most of our deep learning needs, because it’s the most widely adopted,
scalable, and production ready. 
In addition using TensorFlow (or Theano, or CNTK), Keras is able to run on both CPUs and GPUs. 


### <a id="312"></a>Installing Keras


#### <a id="3121"></a>Install useful dependencies

First, we will install useful dependencies:

 * **numpy**, a package which provides support for large, multidimensional arrays and matrices as well as high-level mathematical functions
 * **scipy**, a library used for scientific computation
 * **scikit-learn**, a package used for data exploration
 * **pandas**, a library used for data manipulation and analysis
 
Optionally, it could be useful to install:
 
 * **pillow**, a library useful for image processing
 * **h5py**, a library useful for data serialization used by Keras for model saving
 * **ann_visualizer**, a library which visualize neural network defined using Keras Sequential API

Mentioned dependencies can be installed with single command:

```
pip install numpy scipy scikit-learn pandas pillow h5py
```

#### <a id="3122"></a>Install Theano

Theano can be installed with command:

```
pip install Theano
```

#### <a id="3123"> Install Tensorflow

Note: TensorFlow only supports 64-bit Python 3.5.x on Windows.
Tensorflow can be installed with command:

```
pip install tensorflow
```

Note: this command will install CPU version.
You can find explanation how to install GPU version of the TensorFlow at ...

#### <a id="3124">Install Keras

Keras can be installed with command:

```
pip install keras
```


You can check which version of Keras is installed with the following script:

```
python -c "import keras; print(keras.__version__)"
```

Output on my environment is:
```
2.2.4
```

Also you can upgrade your installation of Keras with:
```
pip install --upgrade keras
```

### <a id="313"></a>Configuring Keras


Keras is configured via json configuration file. 
If you have run Keras at least once, you will find the Keras configuration file at:

```
$HOME/.keras/keras.json
```

NOTE for Windows Users: Please replace **$HOME** with **%USERPROFILE%**.

The default configuration file looks like:
```
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_last"
}
```

Json file consist of the key/value pairs as follows:

 * **epsilon**, type: Float, a numeric fuzzing constant used to avoid dividing by zero in some operations
 * **floatx**, type: String, values: **float16**, **float32**, or **float64**. Default float precision.
 * **backend**, type: String, values: **tensorflow**, **theano**, or **cntk**. Default back-end.
 * **image_data_format**, type: String, values: **channels_last** or **channels_first**. Image data format.

## <a id="32"></a>Develop Your First Neural Network with Keras, XOR gate

In Neural Network [module](https://render.githubusercontent.com/view/ipynb?commit=336ec43a511fd144a1e373f1f3a53feeb9d915ae&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f667261637475732d696f2f6e657572616c2d6e6574776f726b732d7475746f7269616c2f333336656334336135313166643134346131653337336631663361353366656562396439313561652f4e657572616c4e6574776f726b734261736963732e6970796e62&nwo=fractus-io%2Fneural-networks-tutorial&path=NeuralNetworksBasics.ipynb&repository_id=175053175&repository_type=Repository#MultiLayerPerceptronBackpropagationAlgorithm) we showed that problem with XOR gate can't be solved using single layer perceptron. The XOR gate, can be solved with multy layer perceptrons. In that example complex backpropagation algorithm with limited functionality has been implemented directly in source code. 

Keras implements ***complex parts*** of neural networks, like backprop algorithm, activation functions, weights initilaization strategies, loss function etc., so our first example where we will show how to solve XOR gate problem using Keras, will be much simplier comaring to example from Neural Network module.

We need to execute typical steps:

prepare dataset:

```
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
```

define neural network(multi layer perceptron) using Keras Sequential API, with input layer(2 neurons), two hidden layers(8 and 4 neurons, relu as activation fuction) and output layer(1 neuron, sigmoid as activation function):

```
# Adding the input layer and the first hidden layer
model.add(Dense(8, activation = 'relu', input_dim = 2))
# Adding the second hidden layer
model.add(Dense(4, activation = 'relu'))
# Adding the output layer
model.add(Dense(1, activation = 'sigmoid'))
```

compile the model, optimizer is adam, loss is binary crossentropy:

```
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

train a model:

```
model.fit(X, Y, batch_size = 1, epochs = 500)
```

evalute model:

```
scores = model.evaluate(X, Y)
```
Neural network looks like:

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/mlp-xor.jpg "Multy Layer Perceptron - XOR gate")


You can run whole process with command:

```
# from ./xor
python xor.py
```

Result of the learning process is printed in console for each epoch, we can see that after 300 epochs, accuracy of the model is 100%, so the XOR gate problem is solved with Keras with less then 20 lines of code.

### <a id="321"></a>Experiment

Try to develop network with 1 and/or 3 hidden layers, play with number of neurons(do not change size of the input and output neurons), change optimizers e.g. adam, rsmprop, sgd. change number of epochs.
After every change run command:

```
# from ./xor
python xor.py
```

and check results.

## <a id="33"></a>Customer Churn Prediction with Multi Layer Perceptrons

Now we will show more realistic example which can be solved using Multi Layer Perceptrons

Our goal is to predict which customer is going to leave the bank. This problem is usually called ***customer churn prediction***. 
This is a ***binary classiﬁcation problem***, while our network should predict is customer going to leave a bank as 1 or stay as 0. 

As you can see from XOR example, typical steps which needs to be executed are:

1. Load data
2. Prepare data
3. Define Model
4. Compile Model
5. Fit Model
6. Evaluate Model

We will use dataset from bank which contains historical behavior of the customer. Dataset(./src/mlp/Churn_Modelling.csv) has 10000 rows with 14 columns. 

The input variables describes bank customers with following attributes:
 * RowNumber    
 * CustomerId    
 * Surname    
 * CreditScore    
 * Geography    
 * Gender    
 * Age    
 * Tenure    
 * Balance    
 * NumOfProducts    
 * HasCrCard    
 * IsActiveMember    
 * EstimatedSalary    
 * Exited

First few rows from the dataset looks as follows:

```
RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
1,15634602,Hargrave,619,France,Female,42,2,0,1,1,1,101348.88,1
2,15647311,Hill,608,Spain,Female,41,1,83807.86,1,0,1,112542.58,0
3,15619304,Onio,502,France,Female,42,8,159660.8,3,1,0,113931.57,1
4,15701354,Boni,699,France,Female,39,1,0,2,0,0,93826.63,0
5,15737888,Mitchell,850,Spain,Female,43,2,125510.82,1,1,1,79084.1,0
6,15574012,Chu,645,Spain,Male,44,8,113755.78,2,1,0,149756.71,1
```

### <a id="331"></a>Load data

For loading a data we will use Pandas DataFrame which gives us elegant interface for loading.
Dataset is attached to git repo. 

```
dataset = pd.read_csv('Churn_Modelling.csv')
```

### <a id="332"></a>Prepare data

As we already mentioned dataset has 14 columns, but columns **RowNumber** and **CustomerId** are not useful for our analysis, so we will exclude them.
Column **Exited** is our target variable.

```
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values
```

**Country** column has string labels such as **France, Spain, Germany** while **Gender** column has **Male, Female**. 
We have to encode those strings into numeric and we can simply do it using pandas but here library called **ScikitLearn**(strong ML library in Python) are introduced.
We will use **LabelEncoder**. As the name suggests, whenever we pass a variable to this function, this function will automatically encode different labels in that column with values 
between 0 to n_classes-1.

```
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
```

**Gender** is binary, so label encoding is OK, but for column **Country**, label encoding has introduced new problem in our data. 
**LabelEncoder** has replaced **France** with 0, **Germany** 1 and **Spain** 2 but **Germany** is not higher than **France** and **France**
 is not smaller than **Spain** so we need to create a dummy variable for **Country**. Dummy variable is difficult concept, but again we will use **ScikitLearn** library which provides **OneHotEncoder** function.

```
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
```

Now, if you carefully observe data, you will see that data is not scaled properly. 
Some variables has value in thousands while some have value is tens or ones. 
We don’t want any of our variable to dominate on other so let’s go and scale data.
we will use **StandardScaler** from **ScikitLearn** library. 

```
sc = StandardScaler()
X = sc.fit_transform(X)
	
```

Finally data is prepared, so we can move to nedxt step, which is modeling the Neural Network.

### <a id="333"></a>Define Model

Models in Keras are deﬁned as a sequence of layers. 

For our case we will build simple fully-connected Neural Network, with 3 layers, input layer, one hidden layer and output layer.
Such a network is called Multy Layer Perceptron network.

First we create a Sequential model and add layers. 
Fully connected layers are deﬁned using the Dense class, where we need to define following parameters:

First parameter is **output_dim**. It is the number of nodes you want to add to this layer.  
In Neural Network we need to assign weights to each node which is nothing but importance of that node. 
At the time of initialization, weights should be close to 0 and we will randomly initialize weights using **uniform** function. 

For input layer we have to define right number of inputs. 
This can be speciﬁed when creating the ﬁrst layer with the **input_dim**. Remember in our case total number of input variables are 11(14 columns, two columns excluded, last columns is output).
Second layer in our model automatically knows the number of input variables from the first layer, so donćt need to take care about number of inputs for second, third etc. layers.

For first two layers we will use **relu** activation functions, and since we want binary result from output layer, the in last layer we will use **sigmoid** activation function.

Visual represenation of the model

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/mlp-customer-churn-prediction.jpg "Multy Layer Percpeptron - Customer Churn Prediction")

Model Summary

Layer (type)                 Output Shape              Param 
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 72
_________________________________________________________________
dense_2 (Dense)              (None, 6)                 42
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 7
_________________________________________________________________
Total params: 121   
Trainable params: 121   
Non-trainable params: 0   

```
# Adding the input layer and the first hidden layer
model.add(Dense(6, init = 'uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer
model.add(Dense(6, init = 'uniform', activation = 'relu'))
# Adding the output layer
model.add(Dense(1, init = 'uniform', activation = 'sigmoid'))	
```

### <a id="334"></a>Compile Model

Once model is deﬁned, we can compile it. 

Training a network means ﬁnding the best set of weights to make predictions for this problem. 
When compiling, we must specify some properties which are required during training of the network. 

First argument is **Optimizer**, this is algorithm you wanna use to find optimal set of weights.
During model definition phase we already initialized weights, but now we are applying some sort of algorithm which will optimize weights in turn making out Neural Network more powerful. 
This algorithm is **Stochastic Gradient Descent(SGD)**. 
Among several types of **SGD** algorithm the one which we will use is **Adam**. 
If you go in deeper detail of **SGD**, you will find that **SGD** depends on **loss**, thus our second parameter is **loss**. 
Since output variable is binary, we will have to use logarithmic loss function called **binary_crossentropy**.
We want to improve performance of our Neural Network based on accuracy so we add metrics **parameter** as **accuracy**.

```
# compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])	
```

Our Neural Network is now ready to be trained.

### <a id="335"></a>Fit Model

Once model has been defined and compiled, the model is ready for traning. We should give some data to the model and execute the training process. Training of the model is done by calling the ***fit()*** function on the model.

```
# fit the model
model.fit(X, Y, batch_size = 10, epochs = 100)
```

The training process will be executed for a fixed number of iterations through the dataset called
***epochs***, so we must define epochs argument in ***fit()*** function. 

***fit()*** function has much more arguments, but for this example we will define minimum, so in addition to epochs argument, we will define batch_size argument which is number of instances that are evaluated before a weight update in the network is performed 
    
### <a id="336"></a>Evaluate Model

Our model has been trained on the entire dataset, so we can evaluate the performance
of the network on the same dataset. This will only give us an idea of how well we have modeled
the dataset (e.g. train accuracy), but no idea of how well the algorithm might perform on new
data. 

The model can be evalueted usning ***evaluation()*** function on your model and pass it the same input and output used to train the model. This will generate a prediction for each input and output pair and collect scores, including the average loss and any
metrics you have configured, such as accuracy.

```
# evaluate the model
scores = model.evaluate(X, Y)

# print accuracy
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

Since this is still introductionary example same data has been used for traning and evaluation, but in in future we will separate data into train and test datasets for the training and evaluation of your model.

The whole process can be executed with command:

```
# from ./mlp
python mlp.py
```

By running above command, you should see a message for each of the 100 epochs, printing the loss
and accuracy for each, followed by the final evaluation of the trained model on the training dataset.

After 100 epochs we reached accuracy of the approx 86%.
```
Epoch 98/100
10000/10000 [==============================] - 5s 454us/step - loss: 0.3351 - acc: 0.8628
Epoch 99/100
10000/10000 [==============================] - 3s 345us/step - loss: 0.3345 - acc: 0.8614
Epoch 100/100
10000/10000 [==============================] - 3s 324us/step - loss: 0.3341 - acc: 0.8634
```

So we have implemented network which solves real problem of predicting will concrete customer leave or stay in the bank.

### <a id="337"></a>Experiment

Try to change network with adding more layers, play with number of nodes(do not change size of the input and output nodes), change optimizers e.g. adam, rsmprop, sgd., change number of epochs.
After every change run command:

```
# from ./mlp
python mlp.py
```

and check the results.



## <a id="34"></a>Summary

As you can see, during a process of finding appropriate neural network for specific problem, we need to make a lot of 
decision and continuously repeat following steps: 

Idea -> Code -> Experiment 
Idea -> Code -> Experiment 
Idea -> Code -> Experiment 

During mentioned process, we also need to make lot of decision, like:
 * choose variables from dataset(feature engineering)
 * define architecture of the neural network
 * define number of layers
 * define number of neurons in each layer
 * decide on activation function for each layer
 * choose optimization algorithm
 * choose loss function
 * split dataset into training/test/validation sets
 * define batch_size
 * define number of epochs
 * keep a track of the accuracy and loss during training phase
 * evaluate the model
 * save/load a model
 
In a next chapter we will show you how Keras helps us in order to easily implement mentioned steps.

# Deep Learning Models

## <a id="41"></a>Keras Models

Idea behined Keras is to be user friendly, modular, easy to extend.  
The API is ***designed for human beings, not machines,*** and ***follows best practices for reducing cognitive load***.


Neural layers, cost functions, optimizers, initialization schemes, activation functions, and regularization schemes are all standalone modules that you can combine to create a models. New modules are simple to add, as new classes and functions. 

Keras Model API is used to define neural network architecture by defining input layer, hidden layers, output layers, number of neurons in each layer, activation function for each layer etc.

The Model is the core Keras data structure. There are two main types of models available in Keras:
* the Sequential API
* the Functional API

Functional API offers advanced way for defining models.
It allows you to define multiple input or output models as well as models that share layers.

Sequential API is simplier and it will be used in this chapter.

### <a id="412"></a>Sequential API

The easiest way of creating a model in Keras is by using the Sequential API, which lets you stack one layer after the other. 
The problem with the sequential API is that it doesn’t allow models to have multiple inputs or outputs, which are needed for some problems.
Nevertheless, the sequential API is a perfect choice for most problems.

The fundamental data structure in neural networks is the layer. 
***A layer is a data-processing module that takes as input one or more tensors and that outputs one or more tensors.***

Different types of the layers are appropriate for different tensor formats and different types of data processing. 

Vector data, stored in 2D tensors of shape (samples, features), is ussualy processed by densely connected layers, also called fully connected or dense layers (the ***Dense*** class in Keras). 

Image data, stored in 4D tensors, is usually processed by 2D convolution layers (***Conv2D***).

Sequence data, stored in 3D tensors of shape (samples, timesteps, features), is typically processed by recurrent layers such as an ***LSTM*** layer.

Layers are in fact as the LEGO bricks where when are combined together forms a deep learning neural network.

Building deep-learning models in Keras is done by stacking together compatible layers to form data-transformation pipelines.
Every layer will only accept input tensors of a certain shape and will return output tensors of a certain shape. 
Consider the following example:

```
from keras import layers

layer = layers.Dense(16, input_shape=(784,))
```

Layer has been created and it will accept as input 2D tensors where the first dimension is 784. This layer will return a tensor where the first dimension has been transformed to be 16.

Thus this layer can only be connected to a downstream layer that expects 16- dimensional vectors as its input. 

When using Keras, you don’t have to worry about compatibility, because the layers you add to your models are dynamically built to
match the shape of the incoming layer. For instance, suppose you write the following:

```
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, input_shape=(784,)))
model.add(layers.Dense(16))
```
The second layer didn’t receive an input shape argument, instead Keras will automatically inferred its input shape as being the output shape of the layer that came before.

### <a id="413"></a>Layers

Keras Sequential API implements a lot of layers which can be used for different types of neural network(MlP, CNN, LSTM, etc.). Commonly used layers are:
 * Core Layers
    * Dense
    * Dropuout
    * Activation
    * ...

 * Convolutional Layers
    * Conv1D
    * Conv2d
    * ...

 * Pooling Layers
    * MaxPooling1D
    * MaxPooling2D
    * ...

 * Recurrent Layers
    * RNN
    * LSTM
    * ...

Complete list of the implemented layers is described in [Keras documentation](https://keras.io/layers/about-keras-layers/).

### <a id="414"></a>Activation functions

Keras impements neurons activation functions. Activations can either be used through an Activation layer, or through the activation argument supported by all forward layers.

```
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

or 

```
model.add(Dense(64, activation='tanh'))
```

Available [activations functions](https://keras.io/activations/) are:
 * softmax 
 * elu
 * selu
 * softplus
 * softsign
 * relu
 * tanh
 * sigmoid
 * hard_sigmoid
 * exponential
 * linear

 Activation function decides, whether a neuron should be activated or not by calculating weighted sum and further adding bias with it. The purpose of the activation function is to introduce non-linearity into the output of a neuron.

### <a id="415"></a>How to Choose Activation Function 

Both sigmoid and tanh functions are not suitable for hidden layers because if z is very large or very small, the slope of the function becomes very small which slows down the gradient descent which can be visualized in the below video. 

Rectified linear unit (relu) is a preferred choice for all hidden layers because its derivative is 1 as long as z is positive and 0 when z is negative. 

For binary classification, the sigmoid function is a good choice for output layer because the actual output value ‘Y’ is either 0 or 1 so it makes sense for predicted output value to be a number between 0 and 1.

For a multyclass classification softmax activation function is commonly used.

For non-classification problems such as prediction of housing prices, we shall use linear activation function at the output layer only.

### <a id="416"></a>Initializers

Initializations define the way to set the initial random weights of Keras layers.

The keyword arguments used for passing initializers to layers will depend on the layer. Usually it is simply kernel_initializer and bias_initializer:

```
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

The following built-in [initializers](https://keras.io/initializers/) are available as part of the keras.initializers module:

 * Zeros
 * Ones 
 * Constant
 * RandomNormal
 * RandomUniform
 * TruncatedNormal
 * VarianceScaling
 * Orthogonal
 * Identity
 * lecun_uniform
 * glorot_normal
 * glorot_uniform
 * he_normal
 * lecun_normal
 * he_uniform

### <a id="417"></a>Compilations

Once model is defined, and before we start with training, we need to the configure learning process. Again Keras implements "hard part" for us, so method ***compile*** is used for configuration of the learning process. Method receives 3 arguments:

 * optimizer
 * loss function
 * list of the metrics

 Let's see few examples:

```
 # For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

```

### <a id="418"></a>Optimizers

During the training process, we tweak model parameters(weights) and trying to minimize that loss function, in order to make predictions as accurate as it is possible. 

The optimizers are responisible(together with loss function) to update model parameters.

You can think of a hiker trying to get down a mountain with a blindfold on. It’s impossible to know which direction to go in, but there’s one thing she can know: if she’s going down (making progress) or going up (losing progress). Eventually, if she keeps taking steps that lead her downwards, she’ll reach the base.

Similarly, it’s impossible to know what your model’s weights should be right from the start. But with some trial and error based on the loss function (whether the hiker is descending), you can end up getting there eventually.

[ToDo](https://blog.algorithmia.com/wp-content/uploads/2018/05/word-image.png)

Again Keras implements optimizers, like from most Stochastic Gradient Descent, RMSProp, Adam, etc.
When using and optimizer, we should define parameters like learning rate, momentum etc.

You can find details about the optimizers at [link](https://keras.io/optimizers/).

In choosing an optimizer it is important to consider is the network depth, the type of layers and the type of data.

You are free to experiment, but as a hint consider SGD for shallow networks, and either Adam or RMSProp for deepnets.

 
### <a id="419"></a>Loss functions

Loss function is simple method used during training in order to see how well our neural networks models dataset. 

If your predictions are totally off, your loss function will output a higher number. 
If prediction is pretty good, loss function will output a lower number.  As you change pieces of your algorithm to try and improve your model, your loss function will tell you if you’re getting anywhere.

Keras implements loss functions like mean_squared_error, mean_absolute_error, binary_crossentropy, categorical_crossentropy etc.
Details are described [here](https://keras.io/losses/).

Below is a table where you can find a hints how to choose loss and last layer activaton functions.

| Problem                                  | Last layer activation  | Loss                     |  Example                    |
| ---------------------------------------- |:----------------------:| ------------------------:|----------------------------:|
| Binary classification                    | sigmoid                | binary_crossentropy      | dog vs cat                  |
| Multi-class, single-label classification | sigmoid                | categorical_crossentropy | MNIST(10 labels) dog vs cat |
| Multi-class, multi-label classification  | sigmoid                | binary_crossentropy      | News tags classification    |
| Regression to arbitrary values           | none                   | mse                      | house price                 |
| Regression to values between 0 and 1     | sigmoid                | mse                      | 0 is broken, 1 is new       |



### <a id="4110"></a>Training

Model training is executed using ***fit*** function. 

Once the model is compiled, it can be fit, meaning adapt the weights on a training dataset.

Fitting the model requires the training data to be specified. The model is trained using the backpropagation algorithm and optimized according to the optimization algorithm and loss function specified when compiling the model.

The backpropagation algorithm requires that the network be trained for a specified number of epochs or exposures to the training dataset.

Each epoch can be partitioned into groups of input-output pattern pairs called batches. This define the number of patterns that the network is exposed to before the weights are updated within an epoch. It is also an efficiency optimization, ensuring that not too many input patterns are loaded into memory at a time.

A minimal example of fitting a network is:

```
model.fit(X, Y, batch_size = 1, epochs = 500)
```

### <a id="4111"></a>Evaluate

Once the model is trained, it can be evaluated.

The model can be evaluated on the training data, but this will not provide a useful indication of the performance of the network as a predictive model, as it has seen all of this data before.

We can evaluate the performance of the network on a separate dataset, unseen during testing. This will provide an estimate of the performance of the network at making predictions for unseen data in the future.

The model evaluates the loss across all of the test patterns, as well as any other metrics specified when the model was compiled, like classification accuracy. A list of evaluation metrics is returned.

For example, for a model compiled with the accuracy metric, we could evaluate it on a new dataset as follows:

```
model.evaluate(X, Y)
```

### <a id="4112"></a>Predict

Finally, once we are satisfied with the performance of our fit model, we can use that model to make predictions on new data.

For example:

```
model.predict(X)
```

The predictions will be returned in the format provided by the output layer of the network.

In the case of a regression problem, these predictions may be in the format of the problem directly, provided by a linear activation function.

For a binary classification problem, the predictions may be an array of probabilities for the first class that can be converted to a 1 or 0 by rounding.

For a multiclass classification problem, the results may be in the form of an array of probabilities (assuming a one hot encoded output variable) that may need to be converted to a single class output prediction using the argmax function.

## <a id="42"></a>Summary

Keras provides all needed functionality, to cover whole process, from defining neural network, training, evaluating and prediciting until we are satisfied with accuracy of our neural network.

In addition to mentioned functionality, Keras offers more functionality which are helpfull as well, like storing/loading trained models, printing/visualizing model details, optimization techniques to avoid overffiting/underfitting like dropouts, l1 /& l2 regularization, batch norms, callbacks during training etc.

In a next chapter we will show the whole process, how we can for concrete problem, develop a model in order to achieve best possible accuracy. We will start with simple model, then we will build new more advanced models applying regularization tricks in order to end with decent accuracy of the model.


# <a id="51"></a> Model Optimization

## <a id="511"></a> MNIST Handwritten Digits classifier

The MNIST (“NIST” stands for National Institute of Standards and Technology while the “M”
stands for “modified”) dataset is one of the most studied datasets in the computer vision and machine learning literature.

The goal of dataset is to classify the handwritten digits from 0 to 9. 

MNIST itself consists of 60,000 training images and 10,000 testing images. Each image is represented as 784 dim vector, corresponding to the 28x28 grayscale pixel for each image. 
Grayscale pixel intensities are unsigned integers, falling into the range [0;255]. 

All digits are placed on a black background with the foreground being white and shades of gray. Given these raw pixel intensities,
our goal is to train a neural network to correctly classify the digits.

TODO show an image from data set

### <a id="5111"></a> A quick note on **image_data_format**

Before we start, be aware that using TensorFlow, images are represented as NumPy arrays with the shape ***(height, width, depth)***, 
where the depth is the number of channels in the image.

If you are using Theano, images are instead assumed to be represented as ***(depth, height, width)***.

This little difference is the source of a lot of headaches when using Keras.
So, when your model is working with images and if you are getting strange results when using 
Keras(or an error message related to the shape of a given tensor) you should:

 * Check your back-end
 * Ensure your image dimension ordering matches your back-end

### <a id="5112"></a>Load MNIST Dataset

Dataset can be fetched using keras, as follows:

```
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

Inputs training images(variable train_images) has shape (60000, 28, 28), while test images(variable test_images) has shape (10000, 28, 28). 
Train labels(variable train_labels) has shape (10000, 10), while test labels(variable test_labels) has shape (10000, 10).

For a multilayer perceptron inputs will be reshaped from 28 x 28 matrix to 784 vector.

```
trainInputs = trainInputs.reshape(60000, 784)
testInputs = testInputs.reshape(10000, 784)
```

We will scale value of the images from range [0:255] to range [0:1]

```
# covert data to float32
trainInputs.astype('float32')
testInputs.astype('float32')

# scale a values of the data from 0 to 1
trainInputs = trainInputs / 255
testInputs = testInputs / 255
```

Labels will be coverted to vectors:

```
trainLabelsOneHot = to_categorical(train_labels)
testLabelsOneHot = to_categorical(test_labels)
```

## <a id="512"></a> Define & train Model, Multy Layer Perceptron(one layer) 

After dataset has been loaded and prepared, we will define model. Let's start with simple multy layer perceptron with following properties:

 * input layer   
 * hidden layer with 512 neurons, relu as activation function   
 * output layer with 10 neurons(due to 10 output labels) softmax as activation function   


```
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,)))        
model.add(Dense(10, activation='softmax'))
```

Keras offers handy method modelsumary() which prints summary of the model, so summary of our model looks as follows:

_________________________________________________________________
Layer (type)                 Output Shape              Param 
_________________________________________________________________
dense_1 (Dense)              (None, 512)               401920
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130
_________________________________________________________________
Total params: 407,050   
Trainable params: 407,050   
Non-trainable params: 0  
_________________________________________________________________

***Adam*** optimizer will be used(feel free to play with another optimizer like ***rmsprop***).
Due to 10 outputs label, loss function will be ***categorical_crossentropy***.

Compile function will look like:

```
model.compile(optimizer = 'adam', loss= 'categorical_crossentropy',  metrics=['accuracy'])
```

### <a id="5121"></a> Callbacks

TODO ... describe callbacks 

We will train our model with 60 epochs, with ***batch_size*** 256.
Keras fit() function accepts [callbacks](https://keras.io/callbacks/), so for training we will define two callbacks:

 * TimeCallback   
   counts time needed to execute one epoch

 * ModelCheckpoint   
   save the model after every epoch

Implementation of the TimeCallback:
```
import time
import keras

class TimeCallback(keras.callbacks.Callback):
        
    def on_train_begin(self, logs={}):        
        self.times = []    
	
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()
            	
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_start_time)
```

We will define validation dataset which is in fact our dataset consist of test images and test labels.

Keras fit() function returns history object which is collection of the metrics collected after each epoch. History object is also a callback which is automatically registered once fit() method is invoked.

Fit function will look like:

```
# checkpoint callbacks
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                           './outputs/' + utils.getStartTime() + '_' + modelName + '.h5',
                           monitor = 'val_acc',
                           verbose = 1,
                           period = hyper_params.epochs)

# time callback
time_callback = TimeCallback()

history = model.fit(train_data, 
                     train_labels_one_hot,
                     batch_size = 256, 
                     epochs = 60, 
                     verbose = 1, 
                     callbacks = [time_callback, model_checkpoint_callback],
                     validation_data = (test_data, test_labels_one_hot))

```

Now you are ready to train model with command:

```
python mnist.py -m mlp_one_layer
```

After 60 epochs accuracy on test images(val_acc) is 98.31%.
```
Epoch 60/60
60000/60000 [==============================] - 3s 55us/step - loss: 1.0969e-04 - acc: 1.0000 - val_loss: 0.0777 - val_acc: 0.9831
```

### <a id="5122"></a> Evalute trained model

Now we can evaluate trained model:

```
[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
```
evaluate() method computes the loss based on the input you pass it.
Internaly method predicts the output for the given input and then computes the loss function specified in the model.compile method.

After evaluation we are finaly ready for prediction:

```
prediction = model.predict(test_data, hyper_params.batch_size)
```

Method predict() returns numpy array with 10 values(last layer in our model is fully connected layer with 10 neurons), where  each element holds probability for each digit(0-9).

TODO ... classification report ...



Our first loop is completed, ModelCheckpoint callback will save model, so we can use it later on.

So, what now, are we are satified with results, should we consider to try to improve a model ?

One way would be to change some of the hyperpameters in existing model, e.g.
 * new optimizer(for example rsmprop)
 * bacth size
 * number of epochs
 * activation function in hidden layers
 * number of neurons in hidden layers
 * 
or to extend a model with new layers or even develop completely new model.


### <a id="5123"></a> Understand Model Behavior During Training By Plotting History

TODO ... better explanation

If we train a model with change of the hyperparameter, we should be able to compare new results with previous one. 
Therfore we will use history object which is returned by ***fit()*** method and using Python ***matplotlib** library draw graphs which will be stored for analisys.

You can see part of the code bellow, but for full implementetion please, check methods drawGraphByType(), drawAccLossGraph(), drawTimes() in module./src/mnist/utils.py
```

...
plt.style.use("ggplot")
plt.plot(history.history[type])
plt.plot(history.history[valType])    
plt.xlabel('Epoch')
plt.ylabel(type)
plt.legend(['Training ' + str(type) + ' : ' +  str(history.history[type][epochs-1]), 'Validation ' +str(type) + ' : ' +  str(history.history[valType][epochs-1])])
...

```

Model can be trained with command:

```
python mnist.py -m mlp_one_layer
```

So for our very first model after 60 epochs will have on a MNIST dataset we have following graphs:

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/mlp_one_layer_acc_e60.png "MNIST One Layer Perceptron - accuracy after 60 epochs")


![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/mlp_one_layer_loss_e60.png "MNIST One Layer Perceptron - loss after 60 epochs")

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/mlp_one_layer_times_e_60.png "MNIST One Layer Perceptron - times for each epoch")

Are we satified with the results ?
Can we can improve model ?
Let's extend our model with one additional hidden layer.

## <a id="513"></a> Multy Layer Perceptron with two hidden layers

Now we will add hidden second layer, so our model will have:

 * input layer   
 * hidden layer with 512 neurons, relu as activation function   
 * hidden layer with 512 neurons, relu as activation function   
 * output layer with 10 neurons(due to 10 output labels) softmax as activation function   

```
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

```
Summary of the model:

_________________________________________________________________
Layer (type)                 Output Shape              Param 
=================================================================
dense_1 (Dense)              (None, 512)               401920
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5130
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________


Other parameters will remain the same as with MLP model with one hidden layer.

Model can be trained with command:

```
python mnist.py -m mlp_two_layers
```

After 60 epochs accuracy on accuracy on test images(val_acc) is 98.68%, which is slight improvement in comparism to MLP model with one hidden layer.

```
Epoch 60/60
60000/60000 [==============================] - 9s 149us/step - loss: 7.4318e-07 - acc: 1.0000 - val_loss: 0.0991 - val_acc: 0.9859
```


Graphs:

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/mlp_two_layers_acc_e60.png "MNIST Two Layers Perceptron - accuracy after 60 epochs")


![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/mlp_two_layers_loss_e60.png "MNIST Two Layers Perceptron - loss after 60 epochs")

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/mlp_two_layers_times_e_60.png "MNIST Two Layers Perceptron - times for each epoch")


## <a id="514"></a>Convolutional Neural Network

Now we will try to build simple Convolutional neural network(CNN). 
Details about CNN is explained here TODO ...

Model will have:

 * input layer   
 * Convolutional layer with 32 neurons, relu as activation function   
 * MaxPooling layer
 * Convolutional layer with 64 neurons, relu as activation function   
 * MaxPooling layer
 * Convolutional layer with 64 neurons, relu as activation function   
 * Flatten layer
 * Convolutional layer with 64 neurons, relu as activation function  
 * fully connected layer with 64 neurons, relu as activation function  
 * output layer with 10 neurons(due to 10 output labels) softmax as activation function   
 
Keras code for convoluonal model:

```
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1 )))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  

```

Summary of the model:

_________________________________________________________________
Layer (type)                 Output Shape              Param 
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 11, 11, 64)        18496
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 3, 64)          36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 576)               0
_________________________________________________________________
dense_1 (Dense)              (None, 64)                36928
_________________________________________________________________
dense_2 (Dense)              (None, 10)                650
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
_________________________________________________________________


Other parameters will remain the same as with MLP models.

Model can be trained with command:

```
python mnist.py -m conv_net
```

...

After 60 epochs accuracy on test images(val_acc) is 99.34%, which is improvement in comparism to both MLP models.

Graphs:

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_acc_e60.png "CNN - accuracy after 60 epochs")


![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_loss_e60.png "CNN - loss after 60 epochs")

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_times_e_60.png "CNN - times for each epoch")


Please note that to training time simple CNN is almost 2.5 times longer comparing to MLP models due to increased number of neurons and need convolutional steps.



## <a id="515"></a> Techniques to reduce Overfitting

By introducing simple CNN accuracy on test data after 60 epochs is 99,34%, accuracy on a training data is 100%. 
Looks very well so far, but do we have a problem, why we have difference between training and test data set ?

If we closely check accuracy graph on simple CNN, we see gap between training and validation accuracy over the time.

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_acc_e60.png "CNN - accuracy after 60 epochs")

The gap is example of the ***Overfitting***.

Overfitting occurs when you achieve a good fit of your model on the training data, but model does not generalize well on new, unseen data. 
In other words, the model learned patterns specific to the training data, which are irrelevant to unseen data.

In order to reduce overffiting, the technique called regularization is used.
Regularization makes slight modifications to the model such that the model generalizes better. 

This concept improves model’s performance on the unseen.

Keras provides support for regularitaion.

### <a id="5151"></a> L2 & L1 regularization

L1 and L2 are the most common types of regularization. These update the general cost function by adding another term known as the regularization term.

```
Cost function = Loss (say, binary cross entropy) + Regularization term
```

Due to the addition of this regularization term, the values of weight matrices decrease because it assumes that a neural network with smaller weight matrices leads to simpler models. 
Therefore, it will also reduce overfitting to quite an extent.

However, this regularization term differs in L1 and L2.

L2 regularization is also known as weight decay as it forces the weights to decay towards zero (but not exactly zero).

In L1, we penalize the absolute value of the weights. Unlike L2, the weights may be reduced to zero here. 


In Keras, we can directly apply regularization to any layer using the regularizers.

Below is code where we extended simple CNN with L1 and L2 regularizers:

L1

```

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1 ), kernel_regularizer = regularizers.l1(0.0001)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer = regularizers.l1(0.0001)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer = regularizers.l1(0.0001)))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer = regularizers.l1(0.0001)))
model.add(Dense(10, activation='softmax'))    


```

Summary of the model:


Model can be trained with command:

```
python mnist.py -m conv_net_l1
```

...

After 60 epochs accuracy on accuracy on test images(val_acc) is ...

Graphs:

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_l1_acc_e60.png "CNN L1- accuracy after 60 epochs")


![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_l1_loss_e60.png "CNN - L1 loss after 60 epochs")

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_l1_times_e_60.png "CNN L1 - times for each epoch")

L2

```

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1 ), kernel_regularizer = regularizers.l2(0.0001)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer = regularizers.l2(0.0001)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer = regularizers.l2(0.0001)))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer = regularizers.l2(0.0001)))
model.add(Dense(10, activation='softmax'))  


```

Summary of the model:

Model can be trained with command:

```
python mnist.py -m conv_net_l2
```

...

After 60 epochs accuracy on accuracy on test images(val_acc) is ...

Graphs:

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_l2_acc_e60.png "CNN L1- accuracy after 60 epochs")


![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_l2_loss_e60.png "CNN - L1 loss after 60 epochs")

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_l2_times_e_60.png "CNN L1 - times for each epoch")

TODO ... conclusion

### <a id="5152"></a> Batch Normalization

TODO ... describe batch norm...

```

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1 )))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))   


```

Summary of the model:

Model can be trained with command:

```
python mnist.py -m conv_net_batch_norm
```

...

After 60 epochs accuracy on accuracy on test images(val_acc) is ...

Graphs:

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_batch_norm_acc_e60.png "CNN Batch Norm - accuracy after 60 epochs")


![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_batch_norm_loss_e60.png "CNN Batch Norm - loss after 60 epochs")

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_batch_norm_times_e_60.png "CNN Batch Norm - times for each epoch")

TODO ... conclusion

### <a id="5153"></a> Dropouts

This is the one of the most interesting types of regularization techniques. It produces very good results and is consequently the most frequently used regularization technique in the field of deep learning.

So what does dropout do? At every iteration, it randomly selects nerons from neural network and removes them along with all of their incoming and outgoing connections as shown below.

Dropout can be applied to both the hidden layers as well as the input layers.

In Keras, we can implement dropout using the keras core layer. Below is the python code where dropout is applied on multy layer perceptron and on CNN:

* MLP


```

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))   

```

Summary of the model:

TODO ...

Model can be trained with command:

```
python mnist.py -m mlp_two_layers_dropout
```

...

After 60 epochs accuracy on accuracy on test images(val_acc) is ...

Graphs:

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/mlp_two_layers_dropout_e60.png "MNIST Two Layers Perceptron Dropout - accuracy after 60 epochs")


![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/mlp_two_layers_dropout_loss_e60.png "MNIST Two Layers Perceptron Dropout - loss after 60 epochs")

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/mlp_two_layers_dropout_times_e_60.png "MNIST Two Layers Perceptron Dropout - times for each epoch")


* CNN

```

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1 )))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))     


```

Summary of the model:

TODO ...

Model can be trained with command:

```
python mnist.py -m conv_net_dropout
```

...

After 60 epochs accuracy on accuracy on test images(val_acc) is ...

Graphs:

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_dropout_acc_e60.png "CNN Droout - accuracy after 60 epochs")


![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_dropout_loss_e60.png "CNN Dropout - loss after 60 epochs")

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/conv_net_dropout_times_e_60.png "CNN Dropout - times for each epoch")


TODO ... conclusion

### <a id="5154"></a> Data Augmentation


Another powerfull technique to reduce overfitting is to increase the size of the training data. 
Since our dataset is based on imagesd, we can increase the size of the training data by rotating, flipping, scaling, shifting images. 
This technique is known as ***data augmentation***. 

Data augmentation is considered as a mandatory trick in order to improve our predictions.

In Keras, we can perform all of these transformations using ImageDataGenerator. It has a big list of arguments which you you can use to pre-process your training data.

Below is the sample code to implement it.


TODO ... code, graphs, conclusion


## <a id="516"></a> Experiment

TODO ...

## <a id="517"></a> Summary

TODO ...

# Convolutional Neural Networks

## <a id="81"></a> Introduction

In the past, artificial neural networks (ANNs) have proven to be extremely useful for solving problems such as
classification, regression, function estimation and dimensionality reduction. However, it turned out that regular
neural networks were not able to achieve high performances when considering imagery input data, which is the case
when dealing with computer vision problems. 

Convolutional Neural Networks (CNNs) are a special type of neural networks which are especially suited to analyze such imagery input data. This article will provide an overview of the
history of Convolutional Neural Networks, their working principles, and the different building blocks that they are constituted of.

## <a id="82"></a>What are Convolutional Neural Networks and what motives their use?

Classical Multilayer Perceptrons (MLPs) are often fully connected, meaning that every neuron in a particular layer is
connected to all neurons in the previous and subsequent layer. However, such highly connected layers make the
network prone to overfitting because of the many interconnection weights. In addition, the network size would
increase exponentially as the dimensionality of the input data increases. Images are known to be of very high
dimensionality, for example: a simple black-and-white image of 100x100 pixels has a dimensionality of 10 000. This
dimensionality triples when we consider colored RGB images, which is usually the case when dealing with machine
learning applications. In addition, imagery data needs to be flattened into a 1-dimensional vector when inputted in
a multilayer perceptron. However, flattening the data causes spatial information within the image to get lost,
meaning that MLPs treat pixels that are located close together in the same way as pixels that are located further
away from each other.

Because of these limitations, classical MPLs are usually avoided when dealing with imagery data. Ideally, one needs
a way to leverage the spatial correlation of the image pixels such that the network can detect different features
within the image without causing the network to become too large. This can be achieved by making use of a locally
connected network, which is exactly what a convolutional neural network is.

Just as with multilayer-perceptrons, convolutional neural networks consist of different layers that are connected
together. However, in Convolutional Neural Networks, these layers are not fully connected but they are made up of
different filters which are slided over the image. In essence, these filters are a set of cube-shaped weights which are
applied to the image with the purpose of learning the features within it. The different layers within a convolutional
neural network are constituted out of different filters, each with their own specific purpose. The most common layers
that are used within a CNN are the convolutional layer (hence the name Convolutional neural network), the maxpooling layer and the fully connected layer.

## <a id="83"></a>The History of Convolutional Neural Networks

Just like regular neural networks, the concept of convolutional neural networks finds its origin in the human biological
brain. In the early 60’s, it was discovered that the human visual cortex consisted of so-called ‘simple neurons’ and
‘complex neurons’. Both neuron types respond to visual stimuli (i.e., object edges) that are detected in their
individual receptive fields and the combined receptive fields of all neurons cover the whole visual field. However, it
was shown that simple cells react maximally to object edges that have a particular orientation and position within
their receptive field, whereas complex cells are insensitive to the stimuli’s exact position within the receptive field,
often referred to as ‘spatial invariance’. Complex cells achieve this spatial invariance by summing the output of
several simple neurons which have the same stimuli orientations but correspond to different receptive fields. This 
theory, being that complex detectors can be created by summing several simple detectors, is what gave rise to the
development of convolutional neural networks.

The first modern CNN was developed by Yann LeCun in the 1990s, who would later receive the Turing award in 2018
for his contributions in the field of deep learning. LeCun and his team trained a convolutional neural network with
the aim of recognizing hand-written numbers in images. For the purpose of training, he used the MNIST data set
which contains over 50.000 28x28 pixel images of hand-written digits and which is commonly used for training and
testing computer vision applications within the field of machine learning. His research resulted in the development
of ‘LeNet-5’, which is a 7-layered CNN able to recognize hand-written numbers and which was implemented by
several banking institutions with the purpose of detecting the digits on images of bank cheques.
The creation of LeNet, along with the increased use of graphics processing units (GPUs) which allowed faster CNN
implementations, stirred up the research being done in the field of CNNs during the early 2000s. The real
breakthrough of CNNs came with the 2012 ImageNet Challenge, which is an annual computer vision competition in
which teams compete to create the best image classifier using the ImageNet data set. This data set consists of more
than 14 million labeled images from 20.000 categories including animals, vehicles and music instruments.
The winning classifier of the 2012 edition, developed by Alex Krizhevsky, was a large Convolutional Neural Network
called ‘AlexNet’ which achieved the best results ever reported on the ImageNet data set. This achievement drastically
increased the interests in CNNs within the machine learning community and, from then on, the use of Convolutional
Neural Networks has dominated the ImageNet Challenge.

CNNs have enjoyed widespread interest due to their performance in the ImageNet challenge and have since been
used for many computer vision applications. They have proven to perform extraordinary well on other large data sets
such as the CIFAR-100, VisualGenome and MNIST data set, all of which are widely used data sets used for
performance comparison within the field of computer vision. In addition, CNNs have been used extensively for the
purpose of performing facial recognition, analyzing medical images and perform video analysis.

At last, CNNs have also turned out to be useful in other fields than that of computer vision. For example, the use of
CNNs have been explored in the field of Natural Language Processing (NLP), where they have proven to be effective
for standard NLP tasks such as sentence modeling and semantic parsing. Also, CNNs have been implemented for the
purpose of predicting the interactions between different biological proteins and molecules, allowing them to be used
for drug discovery and identifying potential treatment hazards.

## <a id="84"></a>Building blocks of Convolutional Neural Networks

As discussed earlier, convolutional neural networks are constituted out of different layers. Each of these layers
consists of multiple filters and has its own specific purpose. The purpose of training a CNN is to learn the values of
the filter weights, allowing the network to learn the relevant features within the image data. In what follows, the
working principles of the three most common layers (being the convolutional layer, the max-pooling layer and the
fully connected layer) will be discussed.


### <a id="841"></a>Convolutional layers

The convolutional layer consists of a set of cube-shaped filters which are convolved with the input data (or the output
data from the previous layer when the convolutional layer is situated deeper in the network). Each of these filters
has a small width and height (i.e., common heights are 3, 5 or 7 pixels) and a depth that equals the depth of the input
image. This means that the depth of the convolutional filters is equal to 1 when dealing with black-and-white images
and equal to 3 when dealing with RBG (color) images. The different convolutional filters within the layer are slided
over the input data and compute the dot product of the input and the filter, resulting in a so-called convolved feature
map. Note that the width and height of this feature map is lower than that of the input data and depends on the
filter size and the number of pixels the filter is moved every step. However, the depth of the feature map is always
equal to that of the convolutional layer.

The objective of a convolutional layer is to extract features from the input data. CNNs are not limited to one
convolutional layer and, usually, multiple convolutional layers are used within the same network. This allows the
network to learn increasingly more complex feature as the data propagates through the network. For example, the
first convolutional layer is responsible for detecting low-level features (edges and color), whereas subsequent layer
capture higher-level features such as general shapes or objects. This allows the network to gain an increasingly better
understanding of the images in the data set.
	
### <a id="842"></a>Pooling layers
	
Pooling layers are periodically inserted in convolutional networks and are responsible for reducing the spatial size of
the convolved features. The main reason for doing this is to decrease the computational power required to process
the data by means dimensionality reduction.
In general, two types of pooling layers are used, being max pooling layers and average pooling layers. Again, these
layers consist of filters that are slided over the image, resulting in an output with reduced width and height and with
a depth that is equal to that of the input data. The difference between max pooling layers and average pooling layers
is that the output of max pooling layers depends on the maximum pixel values found during the sliding of the filters,
whereas average pooling layers make use of the average pixel values.

### <a id="843"></a>Fully connected layers

Normally, a convolutional neural network consists of a series of convolutional layers and pooling layers which are
able to detect increasingly more complex features within the input image. However, to make the network practically
applicable, the resulting features from these layers need to be interpreted. This is what is being done by the fully
connected layers.
Fully connected layers are added to the end of the network and take as input the flattened vector-representation of
the convolutional and pooling layers. In essence, the fully connected layers represent a regular fully connected neural
network which is trained with the purpose of classifying the object in the input image. Thus, instead of being trained
on the input image directly (as would be the case in classical Multilayer Perceptrons), the network is trained on the
features resulting from the different layers of the convolutional neural network, allowing it to achieve much higher
performance. The output of the fully connected layer is a one-dimensional vector which represents the probabilities
of the input image belonging to a certain class.

An example of a possible CNN architecture with two convolutional layers, two max-pooling layers and a fully
connected layers is represented in the image below. 

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/cnn.jpg "CNN architecure example")


## <a id="85"></a>Summary

Convolutional neural networks have revolutionized the field of computer vision, allowing a machine to mimic the
biological processes that humans use to perceive and interpret imagery data. Applications such as face recognition
and object detection have shifted to the use of convolutional neural networks, and many applications are still to be
discovered. As the theoretical insights in this type of deep neural networks advances, so will the capacity of
computers and machines to perceive their surroundings and to acquire true, human-like sight.

## <a id="86"></a>LeNet MNIST

## <a id="87"></a>Transfer learning




# Recurrent Neural Networks

## <a id="91"></a> Introduction


Artificial Neural Networks are capable of solving a wide range of classification and regression tasks due to the fact
that they are universal approximators. This means that a feed-forward neural network with a finite number of
neurons in its hidden layer can approximate any continuous function with a reasonable accuracy. In addition, this
property can be achieved in a compact way by adding multiple hidden layers to the network, hereby reducing the
number of neurons needed per hidden layer.

Whereas regular multi-layer perceptrons have many advantages over other predictive algorithms, they do not have
the capacity to adequately solve all the problems that are posed within a machine learning context. One of the
problems that has proven to be especially hard to solve for regular neural networks is the sequencing problem.
However, just like convolutional neural networks have proven to achieve much higher accuracies when dealing with
imagery data, it is possible to adapt a multi-layer perceptron in such a way so that it is capable of solving the
sequencing problem. This article will provide an overview of what the sequencing problem is, how neural networks
can be adapted to solve them, and the for what purpose they can be used.

## <a id="92"></a> The Sequencing Problem

Regular neural networks try to find the concept that allows them to perform a mapping between a list of input data
and their corresponding target output. In such static problems, the neural network is trained by using several training
instances without taking into consideration the individual relation between them. In essence, the neural network
assumes that all instances are independent from each other and rejects the order in which instances are presented
to it.

However, many machine learning problems require the analyzation of data in which a relation between individual
training instances can be observed. This is the case when the data that is fed into the network represents a sequence,
hence the name ‘the sequencing problem’. Many examples can be found which represent such data sequence,
including day-to-day air temperatures, the daily closing price of an individual stock or the sequence of words that is
represented by a sentence. From these examples, it is clear to see that valuable information may be hidden in the
order that instances are represented to the network. To capture this information, more complex networks, with
reference to regular multi-layer perceptrons, are needed. This encouraged the invention of the type of neural
networks that is commonly used today for the purpose of dealing with sequential data: Recurrent Neural Networks
(RNNs).

## <a id="93"></a> Recurrent Neural Networks

Recurrent Neural Networks are a type of neural networks that are specifically designed for the purpose of dealing
with sequential data. Conceptionally, they do this by introducing so-called feedback loops into the network’s
architecture, enabling recurrent neural networks to use information from previous calculations in order to determine
the new output. This property gives recurrent neural networks a memory-like capability which allows them to look
back a few steps when dealing with sequential data.

As is the case with regular Neural Networks, RNNs are constituted out of three layers which each have their own
distinct properties and functioning. These layers are the input layer, the hidden layer and the output layer.

1. The Input Layer: The input layer consists out of a set of input neurons which receive the sequential input
data to be processed.

2. The Hidden Layers: The hidden layers, which are constituted out of a finite number of layers, consist out of
a set of neuron layers which are heavily interconnected. In regular fully connected neural networks, all
neurons from a certain layer are connected to all neurons from their adjacent layers. However, the hidden
layers in recurrent neural networks contain temporal loops which allows them to pass their state to future
calculations, effectively serving as the network’s memory.

3. The Output Layer: The output layer consists out of a set of output neurons which together form the
network’s predicted output.

![alt text](https://github.com/fractus-io/keras-tutorial/blob/master/assets/image/rnn.jpg "RNN example")

The weights of a feedforward neural network can easily be trained with regular backpropagation techniques.
However, the use of backpropagation is impracticable and inefficient when dealing with recurrent neural networks
because of the recurrent feedback loops that can be found within them. To solve this problem, a new method had
to be introduced which allowed to propagate the prediction error back through the network, as is done with regular
backpropagation. This led to the introduction of Backpropagation Through Time or BPTT.

Backpropagation Through Time operates as regular backpropagation, with the additional requirement that the
recurrent neural network needs to be unrolled before the technique can be applied. Unrolling of the network is done
by creating copies of the neurons that are linked to a feedback loop within the network. Neurons that have
connection weights to themselves (e.g., to create a memory) can, for example, be represented by two neurons with
the same mathematical properties. The main purpose of performing this unrolling is to create a network that is
acyclic, allowing backpropagation techniques to be used on it.

However, as the depth (i.e., the number of hidden layers) of the network rises, the effectiveness of backpropagation
diminishes. This is causes by the so-called vanishing gradient problem: the magnitude of the gradient decreases
rapidly as it is moved backward through the hidden layers. This results in a situation where neurons that are located
in the first hidden layers learn much more slowly in comparison to neurons that are located at the back of the
network. This problem is solved by introducing a special kind of recurrent neural networks: Long Short-Term Memory
Networks (LSTMs)

## <a id="94"></a> A Special Kind of RNNs: Long Short-Term Memory Networks

Instead of using regular neurons, Long Short-Term Memory networks make us of so-called memory units. These units
are composed out of three gates: an input gate, an output gate and a forget gate. By regulating these gates, the
network is capable of remembering certain values for an arbitrary amount of time. In addition, since Long Short-Term
Memory networks are able to reduce the vanishing gradient problem, they can be used in a deep recurrent network
architecture. This makes them especially well-suited to be used for complex sequencing tasks such as time-series
prediction, speech recognition and semantic parsing.

## <a id="95"></a> Applications of Recurrent Neural Networks

Recurrent Neural Networks, and especially Long Short-Term Memory Networks, have proven to be well-suited for
dealing with complex sequential data. In what follows, an overview will be provided on how recurrent neural
networks are being used within various industries today.

### <a id="951"></a> Energy Forecasting using Time-series prediction

Time-series Forecasting is the branch within machine learning that focusses on predicting parameter values in the
future, based on the parameter values that have been observed in previous timepoints. Using data on the market’s
energy demand during the recent years, recurrent neural networks can be trained in order to provide an estimation
about the market’s energy demand in future time-points. Since a perfect balance between energy supply and energy
demand should always be maintained, having knowledge about the near-future energy demand allows providers to
either increase or decrease production accordingly. This allows energy providers to obtain increased production
efficiencies because of the reduced risk of over-production and better insights into general market trends.

### <a id="952"></a> Speech Recognition using Recurrent Neural Networks

Speech recognition, commonly referred to as voice recognition, is the ability of a machine to recognize humanspoken languages. The technology enhances human-machine interaction and allows one to communicate with
machines without the need of touch-based interaction. Since sentences are made up out of a sequence of words,
recurrent neural networks – which are able to deal with such sequential data – are especially suited for the purpose
of speech recognition. In addition, the fast computational times of recurrent neural networks allow them to perform
speech recognition in real-time, further enhancing the level of interaction.

### <a id="953"></a> Recurrent Neural Networks for the purpose of Natural Language Processing (NLP)

Natural language processing is a field of Artificial Intelligence that focusses on language understanding and language
generation. As discussed earlier, the sequencing problem is inherent to the characteristics of natural languages,
allowing LSTMs to grasp the concepts of words and sentences. This allows the use of LSTMS for the purpose of
estimating the probability distribution of encountering various linguistic units, including words, sentences or even a
whole document.

## <a id="96"></a> Summary

Whereas not obvious at first sight, many machine learning problems involve the analyzation of sequential data which
cannot be solved by regular, fully connected neural networks. Recurrent neural networks, and especially LSTMs, pose
the possibility to analyze such data in a fast way, allowing them to solve complex problems regarding natural
languages and time series. In addition, recurrent neural networks will continue to play an important role within the
field of machine learning and artificial intelligence because of the growing interests in voice-controlled electronics
and speech recognition.



......

#### Early stopping

Early stopping is a kind of cross-validation strategy where we keep one part of the training set as the validation set. When we see that the performance on the validation set is getting worse, we immediately stop the training on the model. This is known as early stopping.

n the above image, we will stop training at the dotted line since after that our model will start overfitting on the training data.

In keras, we can apply early stopping using the callbacks function. Below is the sample code for it.

from keras.callbacks import EarlyStopping

EarlyStopping(monitor='val_err', patience=5)
Here, monitor denotes the quantity that needs to be monitored and ‘val_err’ denotes the validation error.

Patience denotes the number of epochs with no further improvement after which the training will be stopped. For better understanding, let’s take a look at the above image again. After the dotted line, each epoch will result in a higher value of validation error. Therefore, 5 epochs after the dotted line (since our patience is equal to 5), our model will stop because no further improvement is seen.

 


Keras Callbacks

* Understand Model Behavior During Training By Plotting History

* Reduce Overfitting with Dropout Regularization
* Lift Performance with Learning Rate Schedules


...
* underfitting
* Image Augementing
* transfer learning
* typical CNN architectures
* improve nn with hyperparm optimiztion(Keras callbacks)
* RNN/LSTM
* debug Keras models(list layers, show weigths, biases, ...)
* expose keras model via REST
* visualize models
* evaluate models
* cnn calculate layers inputs




 
