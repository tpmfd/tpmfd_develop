The Python scripts in this folder are applied for downscaling the ERA5 reanalysis (2-meter air temperature, 2-meter specific humidity, 10-meter wind speed, surface air pressure) based on a CNN (convolutional neural network)-based model and short-term WRF simulations.

model_train.py
-----------This script is used to train the CNN-based downscaling model with high-resolution WRF simulations.

myconv.py
-----------The customized convolutional layer in the CNN-based downscaling model is given in this script.

model_predict.py
----------- This script is used to downscale ERA5 reanalysis with the trained CNN-based model.
