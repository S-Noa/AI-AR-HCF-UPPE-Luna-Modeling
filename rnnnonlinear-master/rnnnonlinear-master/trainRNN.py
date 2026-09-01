#!/urs/bin/env python3

"""
Train RNN model for predicting higher-order soliton and supercontinuum spectral
and temporal evolutions. Normalized NLSE, GNLSE and multimode GNLSE have been
added.
--salmelal--

Lauri Salmela
lauri.salmela@tuni.fi
Tampere University, 2020
"""

import numpy as np
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.io as sio
import time
import sys

from load_data import *
from make_RNN_model import *
from pred_evo import *
from utils import plot_history


if __name__ == '__main__':

	np.random.seed(123)  # for reproducibility

	print(sys.version)
	start = time.time()

	add_time = 0  # add time stamp for saved results. Yes (1), No (0)

	### training
	num_epoch = 50 # number of epochs
	window_size = 10 # RNN window size

	# select data file
	#filename = 'simulations/HOS_NLSE_time_145.mat' # train_evo=2900, test_evo=100, steps=101, i_x=145
	#filename = 'simulations/HOS_NLSE_spec_126.mat' # train_evo=2900, test_evo=100, steps=101, i_x=145
	#filename = 'simulations/HOS_expt_time_151.mat' # train_evo=2899, test_evo=100, steps=110, i_x=151
	#filename = 'simulations/HOS_expt_spec_126.mat' # train_evo=2899, test_evo=100, steps=110, i_x=126
	#filename = 'simulations/SC_time_276.mat' # train_evo=1250, test_evo=50, steps=200, i_x=276
	filename = 'simulations/SC_spec_251.mat' # train_evo=1250, test_evo=50, steps=200, i_x=251
	#filename = 'simulations/norm_NLSE_time_256.mat' # train_evo=950, test_evo=50, steps=101, i_x=256
	#filename = 'simulations/norm_NLSE_spec_128.mat' # train_evo=950, test_evo=50, steps=101, i_x=128
	#
	#filename = 'simulations/chirped_NLSE_time_256.mat' # train_evo=5900, test_evo=100, steps=101, i_x=256, added_params=10
	#filename = 'simulations/norm_GNLSE_spec_132.mat' # train_evo=11800, test_evo=200, steps=51, i_x=132, added_params=10
	#filename = 'simulations/MMGNLSE_spec_301.mat' # train_evo=950, test_evo=50, steps=100, i_x=256, added_params=25


	# define samples for training and testing, and the number of propagation
	# steps in the evolution
	train_evo, test_evo, steps = 1250, 50, 200

	# define the number of added parameters for chirped_NLSE_time_256 (10),
	# norm_GNLSE_spec_132 (10) and MMGNLSE_spec_301 (25). 0 for other cases.
	added_params = 0

	# load data
	i_x, X_train, X_test, Y_train, Y_test = load_data(filename, train_evo,
													  test_evo, steps,
													  window_size,'dBm') # max/dBm

	# load_data_expt is used with HOS_expt_time_151 and HOS_expt_spec_126
	#i_x, X_train, X_test, Y_train, Y_test = load_data_expt(filename, train_evo,
	#													   test_evo, steps,
	#													   window_size,'max') # max/dBm

	# load_data_addP is used with chirped_NLSE_time_256, norm_GNLSE_spec_132
	# and MMGNLSE_spec_301
	#i_x, X_train, X_test, Y_train, Y_test = load_data_addP(filename, train_evo,
	#													   test_evo, steps,
	#													   window_size,
	#													   added_params, 'maxC') # maxC/dBmQ/dBmCC

	print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
	print("READY...")

	# create new model (0) or load ready model (1)
	model_ready = 0

	if model_ready == 0:
		# Define model architecture
		model = make_RNN_model(window_size, i_x)

		# with additional parameters
		#model = make_RNN_model(window_size, i_x, added_params)

	elif model_ready == 1:
		# Load saved model
		model = load_model('nets/evo.h5')

		model = update_RNN_model(model)

	model.summary()

	### Fit model on training data
	history = model.fit(X_train, Y_train,
						epochs=num_epoch,
						validation_split=0.1,
						verbose=2)

	end = time.time()
	print("Elapsed time: %d seconds." % (end-start))

	timestr = time.strftime("%Y%m%d-%H:%M")

	plot_history(history, timestr, add_time)

	# Save net
	if add_time:
		fname = 'nets/evo_'+timestr+'.h5'
	else:
		fname = 'nets/evo.h5'
	model.save(fname)

	print("TESTING STEP-WISE...")
	Y_submit = model.predict_proba(X_test)

	print('saving results...')
	if add_time:
		fname = 'results/test_results_'+timestr+'.mat'
	else:
		fname = 'results/test_results.mat'
	sio.savemat(fname, {'Y_submit':Y_submit, 'Y_test':Y_test, 'steps':steps,
						'window_size':window_size})

	print("TESTING USING INPUT PROFILE ONLY...")

	Y_submit = pred_evo(model, X_test, test_evo, steps, window_size, i_x)

	# pred_evo_expt is used with HOS_expt_time_151 and HOS_expt_spec_126
	#Y_submit = pred_evo_expt(model, X_test, test_evo, steps, window_size, i_x)

	# pred_evo_expt is used with chirped_NLSE_time_256, norm_GNLSE_spec_132
	# and MMGNLSE_spec_301
	#Y_submit = pred_evo_addP(model, X_test, test_evo, steps, window_size,
	#						 added_params, i_x)

	print('saving results from start...')
	if add_time:
		fname = 'results/full_test_results_'+timestr+'.mat'
	else:
		fname = 'results/full_test_results.mat'
	sio.savemat(fname, {'Y_submit':Y_submit, 'Y_test':Y_test, 'steps':steps,
						'window_size':window_size})
	print('all done')
