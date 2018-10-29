

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
  * [How to install Keras](#312)
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

During our tutorial we will use TensorFlow back-end for most of our, because it’s the most widely adopted,
scalable, and production ready. In addition using TensorFlow (or Theano, or CNTK), Keras is able to run on both
CPUs and GPUs. 


##### <a id="312"></a>How to install Keras

Keras can be installed using **pip**, as follows:

```
$ sudo pip install keras
```

You can which version of Keras is installed using the following script:

```
$ python -c "import keras; print(keras.__version__)"
```
