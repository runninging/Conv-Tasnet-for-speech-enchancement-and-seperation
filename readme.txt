"""
Created on 30/8/2018

@author: kanwang
"""
Function:
    dataset.py: read the data into the model 
    Tasnet_model.py£ºthe forword network
    Tasnet_train.py£ºthe main function to run
    trainer.py£ºcalculate the loss and for training and testing
    utils.py£º process the raw audio and other useful functions
    train.yaml: all the parameters used in the model
    test.py£ºseparate the mixed audio and calculate SDR

Training stage£º
    1.from the beginning: remove the line with "trainer.rerun" in Tasnet_train.py, use "trainer.run" instead
    2.from a trained model: remove the line with "trainer.run" in Tasnet_train.py£¬use "trainer.rerun" instead, and change the "model_path" in train.yaml/temp
    command£ºpython Tasnet_train.py

Testing:
    change the parameters under train.yaml/test
    command£ºpython test.py

model£º
    the trained model

loss£º
    record the process of training

    
