

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
* Multi-Layer Perceptrons
* [Develop Your First Neural Network with Keras](#331)

#### Deep Learning Models
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

#### <a id="31"></a>Introduction To Keras

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research. Kears runs on Python 2.7 or 3.6 and can seamlessly execute on GPUs and CPUs given the underlying frameworks. Keras was initially developed for researchers, with the aim of enabling fast experimentation.

Kerashas is following features:

* Allows easy and fast prototyping (through user friendliness, modularity, and extensibility).
* Supports both convolutional networks and recurrent networks, as well as combinations of the two.
* Runs seamlessly on CPU and GPU.

Keras was developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System), and its primary author and maintainer is François Chollet, a Google engineer.

Keras is based on following guiding principles:
* **User friendliness** Keras is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.

* **Modularity** A model is understood as a sequence or a graph of standalone, fully-configurable modules that can be plugged together with as few restrictions as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions, regularization schemes are all standalone modules that you can combine to create new models.

* **Extensibility** New modules are easy to add (as new classes and functions). 

* **Python** Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.

Keras is distributed under MIT license, meaning that it can be freely used in commercial projects. It's compatable with any version of the Python from 2.7 to 3.6.
Keras has more then 200 000 users, and it is used by Google, Netflix, Uber and hundreds if startups. Keras is also popular framework on Kaggle, the machine-learning competition website, where almost every deep learning competition has been won using Keras models.

##### <a id="311"></a>Keras, TensorFlow, Theano, and CNTK

Keras provides high-level building blocks for developing
deep-learning models. Instead of handling low-level operations such as tensor manipulation, 
Keras relies on a specialized, well-optimized tensor
library to do so, serving as the back-end engine of Keras.

Several different back-end engines can be plugged into Keras. 
Currently, the three existing back-end implementations
are the TensorFlow back-end, the Theano back-end, and the Microsoft Cognitive
Toolkit (CNTK) back-end. Those back-ends are some of the primary platforms for deep learning these days. 

Theano (http://deeplearning.net/software/theano) is developed by the MILA
lab at Université de Montréal, TensorFlow (www.tensorflow.org) is developed by Google,
and CNTK (https://github.com/Microsoft/CNTK) is developed by Microsoft. 

Any piece of code is developed with Keras can be run with any of these back-ends without
having to change anything in the code.

During our tutorial we will use TensorFlow back-end for most of our deep learning needs, because it’s the most widely adopted,
scalable, and production ready. In addition using TensorFlow (or Theano, or CNTK), Keras is able to run on both
CPUs and GPUs. 


##### <a id="312"></a>Installing Keras


###### Install useful dependencies

First, we will install useful dependencies:

 * **numpy**, a package which provides support for large, multidimensional arrays and matrices as well as high-level mathematical functions
 * **scipy**, a library used for scientific computation
 * **scikit-learn**, a package used for data exploration
 * **pandas**, a library used for data manipulation and analysis
 
Optionally, it could be useful to install:
 
 * **pillow**, a library useful for image processing
 * **h5py**, a library useful for data serialization used by Keras for model saving

Mentioned dependencies can be installed with single command:

```
pip install numpy scipy scikit-learn pandas pillow h5py
```

###### Install Theano

Theano can be installed with command:

```
pip install Theano
```

###### Install Tensorflow

Note: TensorFlow only supports 64-bit Python 3.5.x on Windows.
Tensorflow can be installed with command:

```
pip install tensorflow
```

Note: this command will install CPU version.
You can find explanation how to install GPU version of the TensorFlow at ...

###### Install Keras

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

##### <a id="313"></a>Configuring Keras


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

###### A quick note on **image_data_format**

Using TensorFlow, images are represented as NumPy arrays with the shape (height, width, depth), 
where the depth is the number of channels in the image.

However, if you are using Theano, images are instead assumed to be represented as (depth, height, width).

This little nuance is the source of a lot of headaches when using Keras.
So, when your model is working wit images and if you are getting strange results when using 
Keras(or an error message related to the shape of a given tensor) you should:

 * Check your back-end
 * Ensure your image dimension ordering matches your back-end

##### <a id="331"></a>Develop Your First Neural Network with Keras

Developing Neural Network with Keras is straightforward. 
In principle you need to execute following steps:

1. Load data
2. Prepare data
3. Define Model
4. Compile Model
5. Fit Model
6. Evaluate Model

Goal of our Neural Network is to predict customer churn for a certain bank i.e. which customer is going to leave the bank service. 
This is a binary classiﬁcation problem (leave a bank as 1 or stay as 0). 

We will use dataset from bank which contains historical behavior of the customer. Dataset has 10000 rows with 14 columns. 
The input variables describes bank customer with following attributes:
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

###### Load data

For loading a data we will use Pandas DataFrame which gives us elegant interface for loading.
Dataset is attached to git repo. 

```
dataset = pd.read_csv('Churn_Modelling.csv')
```

###### Prepare data

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

###### Define Model

Models in Keras are deﬁned as a sequence of layers. 

For our case we will build simple fully-connected Neural Network, with 3 layers, input layer, one hidden layer and output layer.

First we create a Sequential model and add layers. 
Fully connected layers are deﬁned using the Dense class, where we need to define following parameters:

First parameter is **output_dim**. It is simply the number of nodes you want to add to this layer.  
In Neural Network we need to assign weights to each mode which is nothing but importance of that node. 
At the time of initialization, weights should be close to 0 and we will randomly initialize weights using **uniform** function. 

For input layer we have to define right number of inputs. 
This can be speciﬁed when creating the ﬁrst layer with the **input_dim**. Remember in our case total number of input variables are 11.
Second layer model automatically knows the number of input variables from the first hidden layer.

For first two layers we will use **relu** activation functions, and since we want binary result from output layer, the in last layer we will use **sigmoid** activation function.

```
# Adding the input layer and the first hidden layer
model.add(Dense(6, init = 'uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer
model.add(Dense(6, init = 'uniform', activation = 'relu'))
# Adding the output layer
model.add(Dense(1, init = 'uniform', activation = 'sigmoid'))	
```

###### Compile Model

Now that the model is deﬁned, we can compile it. 
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
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])	
```

Congratulations, you have build your first Deep Learning Neural Network model.
Our Neural Network is now ready to be trained.

