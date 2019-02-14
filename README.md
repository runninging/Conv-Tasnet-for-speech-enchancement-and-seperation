# TASNET: Time-domain Audio Separation Network
Two models are provided here, one with SDR = 11.7 and the other with SDR = 13.6
## Model with SDR = 11.7
You can find the oringnal paper [here](https://arxiv.org/abs/1809.07454). The testing results are shown as follows: ![SDR-11.7](SDR-11.7.png)
### Function
- dataset.py: read the data into the model 
- Tasnet_model.py：the forword network
- Tasnet_train.py：the main function to run
- trainer.py：calculate the loss and for training and testing
- utils.py： process the raw audio and other useful functions
- train.yaml: all the parameters used in the model
- test.py：separate the mixed audio and calculate SDR
### Training stage：
- from the beginning: remove the line with "trainer.rerun" in Tasnet_train.py, use "trainer.run" instead
- from a trained model: remove the line with "trainer.run" in Tasnet_train.py，use "trainer.rerun" instead, and change the "model_path" in train.yaml/temp
## Model with SDR = 13.6
You can find the oringnal paper [here](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/2290.html).
### Function
- Tasnet_model_13.6.py: the forword network of SDR = 13.6 system

The only diffierence between 11.7 and 13.6 system is the forward network, just replace the Tasnet_model.py with Tasnet_model_13.6.py and run.
