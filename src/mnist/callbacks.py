'''

 Copyright (c) 2019 Fractus IT d.o.o. <http://fractus.io>
 
'''

import time
import keras


class TimeCallback(keras.callbacks.Callback):
        
    def on_train_begin(self, logs={}):        
        self.times = []    
	
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()
            	
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_start_time)
        
    

        

