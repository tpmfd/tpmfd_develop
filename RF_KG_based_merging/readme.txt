The Python scripts in this folder are applied for merging the downscaled meteorological data (2-meter air temperature, 2-meter specific humidity, 10-meter wind speed, surface air pressure) with in-situ observations.

RF_train.py
------------
This script is used to train the random forest (RF) model based on the downscaled meteorological data and in-situ observations.

RF_predict.py
--------------
This script is used to generated high-resolution meteorological data based on the trained RF model.

kriging_adjust.py
------------------
This script is applied for correcting the residuals of the RF prediction, using the Ordinary Kriging.
