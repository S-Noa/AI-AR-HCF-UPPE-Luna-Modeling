RNNnonlinear
========================================================================

`RNNnonlinear` contains codes to create the LSTM neural networks and codes for visualization of the model-free prediction results of ultrafast optical nonlinear dynamics.
The codes have been written on Python 3.6 using Keras (v2.2.0) with TensorFlow backend (v1.9.0). Matlab have been used for the visualisation of the results.


Scripts
--------------

``trainRNN.py``
The main script that can be used for creating and training the neural network. It also tests the network on a subset of unseen data.

``testRNN.py``
Script for testing an existing neural network.

``make_RNN_model.py``
Set the architecture and hyperparameters of the neural network.

``pred_evo.py``
Predict the evolution for various input intensity profiles. Notice the separate functions used with 1) "HOS_expt_time_151" and "HOS_expt_spec_126" and 2) "chirped_NLSE_time_256", "norm_GNLSE_spec_132" and "MMGNLSE_spec_301" data sets.

``load_data.py``
Code for loading and preprocessing the training and test data from "simulations" folder. Notice the separate functions used with 1) "HOS_expt_time_151" and "HOS_expt_spec_126" and 2) "chirped_NLSE_time_256", "norm_GNLSE_spec_132" and "MMGNLSE_spec_301" data sets.


plot_results
--------------

This folder contains Matlab codes to plot the results shown in the manuscript.

results
--------------

This folder (~0.7 GB) contains the full sets of results that can be plotted using the codes in 'plot_results'.

simulations
--------------

The full simulation data sets (~6 GB) used for the training and testing can be downloaded from here: https://zenodo.org/deposit/4304771.

nets
--------------

This folder contains the neural networks used for producing the results. The training process is described below.


Training procedure
--------------

The architecture and training process of the RNNs used here is described in this section. All networks consist of a LSTM layer, two dense layers (ReLU activation) and an output layer (sigmoid activation). All the networks were trained with "RMSprop" optimiser and "mean squared error" loss function.

`HOS_NLSE_time_145_60e`
The network layer sizes are 161, 161, 161, and 145, for the LSTM, two hidden and the output layer, respectively, and the network was trained 30+30 epochs with learning rates of 1e-4 and 1e-5.

`HOS_NLSE_spec_126_80e`
The network layer sizes are 161, 161, 161, and 126, for the LSTM, two hidden and the output layer, respectively, and the network was trained 50+30 epochs with learning rates of 1e-4 and 1e-5. The network has been trained with data in logarithmic scale.

`HOS_NLSE_spec_lin_126_120e`
The network layer sizes are 161, 161, 161, and 126, for the LSTM, two hidden and the output layer, respectively, and the network was trained 60+60 epochs with learning rates of 1e-4 and 1e-5. The network has been trained with data in linear scale.

`HOS_expt_time_151_60e`
The network layer sizes are 161, 161, 161, and 151, for the LSTM, two hidden and the output layer, respectively, and the network was trained 30+30 epochs with learning rates of 1e-4 and 1e-5.

`HOS_expt_spec_126_80e`
The network layer sizes are 161, 161, 161, and 125, for the LSTM, two hidden and the output layer, respectively, and the network was trained 50+30 epochs with learning rates of 1e-4 and 1e-5. The network has been trained with data in logarithmic scale.

`HOS_expt_spec_lin_126_120e`
The network layer sizes are 161, 161, 161, and 125, for the LSTM, two hidden and the output layer, respectively, and the network was trained 60+60 epochs with learning rates of 1e-4 and 1e-5. The network has been trained with data in linear scale.

`SC_time_276_120e`
The network layer sizes are 350, 350, 350, and 276, for the LSTM, two hidden and the output layer, respectively, and the network was trained 60+60 epochs with learning rates of 1e-4 and 1e-5.

`SC_spec_251_80e`
The network layer sizes are 250, 250, 250, and 251, for the LSTM, two hidden and the output layer, respectively, and the network was trained 50+30 epochs with learning rates of 1e-4 and 1e-5. The network has been trained with data in logarithmic scale.

`SC_spec_lin_251_100e`
The network layer sizes are 250, 250, 250, and 251, for the LSTM, two hidden and the output layer, respectively, and the network was trained 50+50 epochs with learning rates of 1e-4 and 1e-5. The network has been trained with data in linear scale.

`norm_NLSE_time_256_80e`
The network layer sizes are 300, 300, 300, and 256, for the LSTM, two hidden and the output layer, respectively, and the network was trained 50+30 epochs with learning rates of 1e-4 and 1e-5.

`norm_NLSE_spec_128_80e`
The network layer sizes are 250, 250, 250, and 128, for the LSTM, two hidden and the output layer, respectively, and the network was trained 50+30 epochs with learning rates of 1e-4 and 1e-5. The network has been trained with data in logarithmic scale.

`chirped_NLSE_time_256_80e`
The network layer sizes are 300, 300, 300, and 256, for the LSTM, two hidden and the output layer, respectively, and the network was trained 50+30 epochs with learning rates of 1e-4 and 1e-5.

`norm_GNLSE_spec_132_80e`
The network layer sizes are 265, 265, 265, and 132, for the LSTM, two hidden and the output layer, respectively, and the network was trained 50+30 epochs with learning rates of 1e-4 and 1e-5. The network has been trained with data in logarithmic scale.

`MMGNLSE_spec_301_80e`
The network layer sizes are 500, 500, 500, and 301, for the LSTM, two hidden and the output layer, respectively, and the network was trained 50+30 epochs with learning rates of 1e-4 and 1e-5. The network has been trained with data in logarithmic scale.
