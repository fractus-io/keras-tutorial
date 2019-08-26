

# Keras Tutorial

#### Table of Content

#### Introduction
* [Introduction To Deep Learning](#1)
* Biologicals
* Artificial Neural Networks - Forward Propagation

#### Artificial Neural Networks
* Gradient Descent
* Backpropagation
* Vanishing Gradient
* Activation Functions

#### Deep Learning with Keras
* [Introduction to Keras](#31)
  * [Keras, TensorFlow, Theano, and CNTK](#311)
  * [Installing Keras](#312)
  * [Configuring Keras](#313)
* [Develop Your First Neural Network with Keras, XOR gate](#321)
* [Multi Layer Perceptrons](#331)
* [Summary](#341)


#### Deep Learning Models
* [Keras Models](#41)
* Shallow and Deep Neural Networks
* Model Architectures
* Train the Model
* Serialize the Model

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

## <a id="31"></a>Introduction To Keras

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. 
It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to 
doing good research. Kears runs on Python 2.7 or 3.6 and can seamlessly execute on GPUs and CPUs given the underlying frameworks. 
Keras was initially developed for researchers, with the aim of enabling fast experimentation.

Keras has following features:

* Allows easy and fast prototyping (through user friendliness, modularity, and extensibility).
* Supports both convolutional networks and recurrent networks, as well as combinations of the two.
* Runs seamlessly on CPU and GPU.

Keras was developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System), 
and its primary author and maintainer is François Chollet, a Google engineer.

Keras is based on following guiding principles:
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


#### Install useful dependencies

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

#### Install Theano

Theano can be installed with command:

```
pip install Theano
```

#### Install Tensorflow

Note: TensorFlow only supports 64-bit Python 3.5.x on Windows.
Tensorflow can be installed with command:

```
pip install tensorflow
```

Note: this command will install CPU version.
You can find explanation how to install GPU version of the TensorFlow at ...

#### Install Keras

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

#### A quick note on **image_data_format**

Using TensorFlow, images are represented as NumPy arrays with the shape (height, width, depth), 
where the depth is the number of channels in the image.

However, if you are using Theano, images are instead assumed to be represented as (depth, height, width).

This little nuance is the source of a lot of headaches when using Keras.
So, when your model is working wit images and if you are getting strange results when using 
Keras(or an error message related to the shape of a given tensor) you should:

 * Check your back-end
 * Ensure your image dimension ordering matches your back-end


### <a id="321"></a>Develop Your First Neural Network with Keras, XOR gate

In Neural Network [module](https://render.githubusercontent.com/view/ipynb?commit=336ec43a511fd144a1e373f1f3a53feeb9d915ae&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f667261637475732d696f2f6e657572616c2d6e6574776f726b732d7475746f7269616c2f333336656334336135313166643134346131653337336631663361353366656562396439313561652f4e657572616c4e6574776f726b734261736963732e6970796e62&nwo=fractus-io%2Fneural-networks-tutorial&path=NeuralNetworksBasics.ipynb&repository_id=175053175&repository_type=Repository#MultiLayerPerceptronBackpropagationAlgorithm) we showed that problem with XOR gate can't be solved using single layer perceptron. The XOR gate, can be solved with multy layer perceptrons. In that example complex backpropagation algorithm with limited functionality has been implemented directly in source code. 

Keras implements ***complex parts*** of neural networks, like backprop algorithm, various activation functions, weights initilaization strategies, loss function etc.,  so in our first example we will show how we can solve XOR gate problem using Keras. 

We need to:

prepare dataset:

```
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
```

define neural network using Keras Sequential API:

```
# Adding the input layer and the first hidden layer
model.add(Dense(8, activation = 'relu', input_dim = 2))
# Adding the second hidden layer
model.add(Dense(4, activation = 'relu'))
# Adding the output layer
model.add(Dense(1, activation = 'sigmoid'))
```

compile the model:

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

Already after 300 epochs, accuracy of the model is 100%, so the XOR gate problem is solved with Keras with less then 20 line sof code.


You can run whole process with command:

```
# from ./xor
python xor.py
```


### <a id="331"></a>Multi Layer Perceptrons

Now we will show more realistic example which can be solved using Multi Layer Perceptrons

Our goal is to predict which customer is going to leave the bank, this problem is usually called ***customer churn prediction***. 
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

#### Load data

For loading a data we will use Pandas DataFrame which gives us elegant interface for loading.
Dataset is attached to git repo. 

```
dataset = pd.read_csv('Churn_Modelling.csv')
```

#### Prepare data

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

Finally data are prepared, so we can start to model our Neural Network.

#### Define Model

Models in Keras are deﬁned as a sequence of layers. 

For our case we will build simple fully-connected Neural Network, with 3 layers, input layer, one hidden layer and output layer.
Such a network is called Multy Layer Perceptron network.

First we create a Sequential model and add layers. 
Fully connected layers are deﬁned using the Dense class, where we need to define following parameters:

First parameter is **output_dim**. It is simply the number of nodes you want to add to this layer.  
In Neural Network we need to assign weights to each mode which is nothing but importance of that node. 
At the time of initialization, weights should be close to 0 and we will randomly initialize weights using **uniform** function. 

For input layer we have to define right number of inputs. 
This can be speciﬁed when creating the ﬁrst layer with the **input_dim**. Remember in our case total number of input variables are 11.
Second layer model automatically knows the number of input variables from the first hidden layer.

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

#### Compile Model

Now model is deﬁned, so we can compile it. 
Compiling the model uses the eﬃcient numerical libraries under the covers (the so-called back-end) such as **Theano** or **TensorFlow**. 
The back-end automatically chooses the best way to represent the network for training and making predictions to run on your hardware. 

Training a network means ﬁnding the best set of weights to make predictions for this problem. 
When compiling, we must specify some  properties required when training the network. 

First argument is **Optimizer**, this is nothing but the algorithm you wanna use to find optimal set of weights.
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

Congratulations, you have build your first Deep Learning Neural Network model.
Our Neural Network is now ready to be trained.

#### Fit Model

Once our model has been defined and compiled, the model is ready for traning. We should give some data to the model and executethe training process. Training of the model is done by calling the ***fit()*** function on the model.

```
# fit the model
model.fit(X, Y, batch_size = 10, epochs = 100)
```

The training process will be executed for a fixed number of iterations through the dataset called
***epochs***, so we must define epochs argument in ***fit()*** function. 

***fit()*** function has much more arguments, but for this example we will define minimum, so in addition to epochs argument, we will define batch_size argument which io  number of instances that are evaluated before a weight update in the network is performed 
    
#### Evaluate Model

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

Since this is very first example same data has been used for , so in future we will separate data into train and
test datasets for the training and evaluation of your model.

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

Note: feel free to experiment, try change neural network architecture, change other parameters and see can you improve accuracy or the model.



### <a id="341"></a>Summary

As you can see, during a process of finding appropriate neural network for specific problem, we need to make a lot of 
decision and to continuously loop over the steps: 

Idea -> Code -> Experiment 

So, we need to:
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

## >Deep Learning Models

### <a id="41"></a>Keras Models

Idea behined Keras is to be user friendly, modular, easy to extend. The API is ***designed for human beings, not machines,*** and ***follows best practices for reducing cognitive load***.


Neural layers, cost functions, optimizers, initialization schemes, activation functions, and regularization schemes are all standalone modules that you can combine to create a models. New modules are simple to add, as new classes and functions. 

Keras Model API is used define neural network architecture by defining input layer, hidden layers, output layers, number of neurons in each layer, activation function for each layer etc.

The Model is the core Keras data structure. There are two main types of models available in Keras:
* the Sequential API
* the Functional API

#### Sequential API

The easiest way of creating a model in Keras is by using the sequential API, which lets you stack one layer after the other. 
The problem with the sequential API is that it doesn’t allow models to have multiple inputs or outputs, which are needed for some problems.
Nevertheless, the sequential API is a perfect choice for most problems.

The fundamental data structure in neural networks is the layer. 
***A layer is a data-processing module that takes as input one or more tensors and that outputs one or more tensors.***

Different type of the layers are appropriate for different tensor formats and different types of data processing. 

Vector data, stored in 2D tensors of shape (samples, features), is ussualy processed by densely connected layers, also called fully connected or dense layers (the Dense class in Keras). 

Image data, stored in 4D tensors, is usually processed by 2D convolution layers (Conv2D).

Sequence data, stored in 3D tensors of shape (samples, timesteps, features), is typically processed by recurrent layers such as an LSTM layer.

Layers are in fact as the LEGO bricks where when are combined together forms a deep learning neural network.

Building deep-learning models in Keras is done by stacking together compatible layers to form data-transformation pipelines.
Every layer will only accept input tensors of a certain shape and will return output tensors of a certain shape. 
Consider the following example:

```
from keras import layers
layer = layers.Dense(16, input_shape=(784,))
```

Layer has been created and it will only accept as input 2D tensors where the first dimension is 784. This layer will return a tensor where the first dimension has been transformed to be 16.

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
The second layer didn’t receive an input shape argumen, instead Keras will automatically inferred its input shape as being the output shape of the layer that came before.





Picking the right network architecture is more an art than a science; and although
there are some best practices and principles you can rely on, only practice can help
you become a proper neural-network architect. 





 
