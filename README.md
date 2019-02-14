# Conv-TASNET:
## Model with SDR = 16.7 (15.0 in the paper)
You can find the oringnal paper [here](https://arxiv.org/abs/1809.07454). The testing results are shown as follows: ![SDR-11.7](SDR-16.7.png)
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
