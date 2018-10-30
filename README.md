

# Keras Tutorial

#### Table of Content

#### Introduction
* [Introduction To Deep Learning](#1)
* Biological Neural Networks
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
* Develop Your First Neural Network with Keras

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
 
Optionally, it could be useful to install:
 
 * **pillow**, a library useful for image processing
 * **h5py**, a library useful for data serialization used by Keras for model saving

Mentioned dependencies can be installed with single command:

```
pip install numpy scipy scikit-learn pillow h5py
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


You can which version of Keras is installed using the following script:

```
python -c "import keras; print(keras.__version__)"
```

Output on my environment will be
```
2.2.4
```

Also you can upgrade your installation of Keras using command:
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

The default configuration file looks like this:
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
