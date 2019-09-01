'''

 Copyright (c) 2019 Fractus IT d.o.o. <http://fractus.io>
 
'''
import utils
import unittest
import numpy as np
from keras.datasets import mnist



class TestUtils(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_load_prepareData(self):

        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
        # check loaded dataset    
        self.assertEqual(len(train_images.shape), 3)        
        self.assertEqual(train_images.shape[0], 60000)
        self.assertEqual(train_images.shape[1], 28)
        self.assertEqual(train_images.shape[2], 28)

        self.assertEqual(len(test_images.shape), 3)        
        self.assertEqual(test_images.shape[0], 10000)
        self.assertEqual(test_images.shape[1], 28)
        self.assertEqual(test_images.shape[2], 28)
        
        self.assertEqual(len(train_labels.shape), 1)
        self.assertEqual(train_labels.shape[0], 60000)

        self.assertEqual(len(test_labels.shape), 1)
        self.assertEqual(test_labels.shape[0], 10000)
       
        # unique number of test_labels
        self.assertEqual(len(np.unique(test_labels)), 10) 
                               
        # prepare data
        (train_data, train_labels_one_hot), (test_data, test_labels_one_hot) = utils.prepareData(train_images, train_labels, test_images, test_labels, 'False')

        self.assertEqual(len(train_data.shape), 2)        
        self.assertEqual(train_data.shape[0], 60000)
        self.assertEqual(train_data.shape[1], 784)
        
        self.assertEqual(train_data.dtype, 'float64')
        self.assertEqual(test_data.dtype, 'float64')

        self.assertEqual(len(test_data.shape), 2)        
        self.assertEqual(test_data.shape[0], 10000)
        self.assertEqual(test_data.shape[1], 784)

        self.assertEqual(len(train_labels_one_hot.shape), 2)        
        self.assertEqual(train_labels_one_hot.shape[0], 60000)
        self.assertEqual(train_labels_one_hot.shape[1], 10)

        # prepare data
        (train_data, train_labels_one_hot), (test_data, test_labels_one_hot) = utils.prepareData(train_images, train_labels, test_images, test_labels, 'True')

        self.assertEqual(len(train_data.shape), 4)        
        self.assertEqual(train_data.shape[0], 60000)
        self.assertEqual(train_data.shape[1], 28)
        self.assertEqual(train_data.shape[2], 28)
        self.assertEqual(train_data.shape[3], 1)

        self.assertEqual(train_data.dtype, 'float64')
        self.assertEqual(test_data.dtype, 'float64')

        self.assertEqual(len(test_data.shape), 4)        
        self.assertEqual(test_data.shape[0], 10000)
        self.assertEqual(test_data.shape[1], 28)
        self.assertEqual(test_data.shape[2], 28)
        self.assertEqual(test_data.shape[3], 1)

        self.assertEqual(len(train_labels_one_hot.shape), 2)        
        self.assertEqual(train_labels_one_hot.shape[0], 60000)
        self.assertEqual(train_labels_one_hot.shape[1], 10)
        
if __name__ == '__main__':
    unittest.main()
    
    
