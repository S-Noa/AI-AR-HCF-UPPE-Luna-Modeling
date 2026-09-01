#!/urs/bin/env python3

"""
Codes for loading and preprocessing the data.
"""

import numpy as np
import scipy.io as sio


def load_data(filename, train_evo, test_evo, steps, window_size,
              normalization='none'):
    """
    Load data. Add input pulse profile as the first step.
    The input for the network is 'window_size' times the input profile.

    Parameters
    ----------
    filename : filename as string
    train_evo : number of training evolutions as integer
    test_evo : number of test evolutions as integer
    steps : number of propagation steps as integer
    window_size : RNN window size as integer
    normalization : none (default), max, dBm, manual ...

    Returns
    -------
    i_x : number of grid points (spectral/temporal) as integer
    X_train : training data input, shape (N, window_size, i_x)
    X_test : testing data input, shape (M, window_size, i_x)
    Y_train : training data output, shape (N, i_x)
    Y_test : testing data output, shape (M, i_x)
    """

    # Load data
    mat_contents = sio.loadmat(filename)
    data = mat_contents['data']
    print("data loaded...")
    print(data.shape)

    if normalization == 'none':
        pass
        
    elif normalization == 'max':  # linear scale (normalized)
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        data = data/m_max

    elif normalization == 'dBm':  # logarithmic scale
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        data=data/m_max  # normalize
        data = 10*np.log10(data)  # dB scale
        dBlim = 55  # define dynamic range
        data[data < -dBlim] = -dBlim  # set spectrum <-55 to -55
        data = data/dBlim + 1

    elif normalization == 'manual':
        m_max = 10369993.175721595 # SC spectral domain
        print('max:', m_max)
        data = data/m_max


    # the number of grid points
    i_x = data.shape[1]

    # Make the time series
    num_evo = train_evo + test_evo
    evo_size = steps - 1
    num_samples = np.round(num_evo*evo_size).astype(int)
    X_data_series = np.zeros((num_samples, window_size, i_x))
    Y_data_series = np.zeros((num_samples, i_x))

    for evo in range(num_evo):
        evo_data = np.transpose(data[evo, :, :])

        # tile the beginning of the evolution with 'window_size' input profiles
        temp1 = evo_data[0, :]
        temp2 = np.tile(temp1, (window_size - 1, 1))
        evo_data = np.vstack((temp2, evo_data))

        for step in range(evo_size):
            input_data = evo_data[step:step + window_size, :]
            output_data = evo_data[step + window_size, :]
            series_idx = evo*evo_size + step
            X_data_series[series_idx, :, :] = input_data
            Y_data_series[series_idx, :] = output_data


    X_train = X_data_series[:num_samples - test_evo*evo_size]
    X_test = X_data_series[num_samples - test_evo*evo_size:]
    Y_train = Y_data_series[:num_samples - test_evo*evo_size]
    Y_test = Y_data_series[num_samples - test_evo*evo_size:]

    return i_x, X_train, X_test, Y_train, Y_test


def load_data_expt(filename, train_evo, test_evo, steps, window_size,
                   normalization='none'):
    """
    Load data to predict the evolution from a given 'window_size' steps.
    To be used with data sets "HOS_expt_time_151" and "HOS_expt_spec_126".

    Parameters
    ----------
    filename : filename as string
    train_evo : number of training evolutions as integer
    test_evo : number of test evolutions as integer
    steps : number of propagation steps as integer
    window_size : RNN window size as integer
    normalization : none (default), max, manual ...

    Returns
    -------
    i_x : number of grid points (spectral/temporal) as integer
    X_train : training data input, shape (N, window_size, i_x)
    X_test : testing data input, shape (M, window_size, i_x)
    Y_train : training data output, shape (N, i_x)
    Y_test : testing data output, shape (M, i_x)
    """

    # Load data
    mat_contents = sio.loadmat(filename)
    data = mat_contents['data']
    print("data loaded...")
    print(data.shape)

    if normalization == 'none':
        pass

    elif normalization == 'max':  # linear scale (normalized)
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        data = data/m_max

    elif normalization == 'dBm':  # logarithmic scale
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        data=data/m_max  # normalize
        data = 10*np.log10(data)  # dB scale
        dBlim = 55  # define dynamic range
        data[data < -dBlim] = -dBlim  # set spectrum <-55 to -55
        data = data/dBlim + 1

    elif normalization == 'manual':
        m_max = 10369993.175721595 # SC spectral domain
        print('max:', m_max)
        data = data/m_max

    # the number of grid points
    i_x = data.shape[1]

    # Make the time series
    num_evo = train_evo+test_evo
    evo_size = steps-window_size
    num_samples = np.round(num_evo*evo_size).astype(int)
    X_data_series = np.zeros((num_samples, window_size, i_x))
    Y_data_series = np.zeros((num_samples, i_x))

    for evo in range(num_evo):
        for step in range(evo_size):
            input_data = np.transpose(data[evo, :, step:step + window_size])
            output_data = data[evo, :, step + window_size]
            series_idx = evo*evo_size + step
            X_data_series[series_idx,:,:] = input_data
            Y_data_series[series_idx,:] = output_data

    X_train = X_data_series[:num_samples - test_evo*evo_size]
    X_test = X_data_series[num_samples - test_evo*evo_size:]
    Y_train = Y_data_series[:num_samples - test_evo*evo_size]
    Y_test = Y_data_series[num_samples - test_evo*evo_size:]

    return i_x, X_train, X_test, Y_train, Y_test


def load_data_addP(filename, train_evo, test_evo, steps, window_size,
                   added_params, normalization='none'):
    """
    Load data. Add input pulse profile as the first step.
    The input for the network is 'window_size' times the input profile.
    The 'added_params' is removed from the network output.

    Parameters
    ----------
    filename : filename as string
    train_evo : number of training evolutions as integer
    test_evo : number of test evolutions as integer
    steps : number of propagation steps as integer
    window_size : RNN window size as integer
    added_params : number of additional parameters as integer
    normalization : none (default), max, maxC, dBmQ, dBmCC, manual ...

    Returns
    -------
    i_x : number of grid points (spectral/temporal) as integer
    X_train : training data input, shape (N, window_size, i_x+added_params)
    X_test : testing data input, shape (M, window_size, i_x+added_params)
    Y_train : training data output, shape (N, i_x)
    Y_test : testing data output, shape (M, i_x)
    """

    # Load data
    mat_contents = sio.loadmat(filename)
    data = mat_contents['data']
    print("data loaded...")
    print(data.shape)

    if normalization == 'none':
        pass

    elif normalization == 'max':  # linear scale (normalized)
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        data = data/m_max

    elif normalization == 'maxC': # linear scale with chirp parameter
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        # normalize data
        data[:, added_params:, :] = data[:, added_params:, :]/m_max
        # normalize additional parameters (interval [-8,8] for chirp)
        data[:, :added_params, :] = (data[:, :added_params, :] + 8)/16

    elif normalization == 'dBmQ':  # logarithmic scale with q parameter
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        # normalize data
        data[:, added_params:, :] = data[:, added_params:, :]/m_max
        data[:, added_params:, :] = 10*np.log10(data[:, added_params:, :]) # dB scale
        dBlim = 55  # define dynamic range
        data[data < -dBlim] = -dBlim  # set spectrum <-55 to -55
        data[:, added_params:, :] = data[:, added_params:, :]/dBlim + 1
        # use interval [1,9] for q parameter
        data[:, :added_params, :] = data[:, :added_params, :]/9

    elif normalization == 'dBmCC':  # logarithmic scale with coupling conditions
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        # normalize data, coupling conditions are already normalized ([0, 1])
        data[:, added_params:, :] = data[:, added_params:, :]/m_max
        data[:, added_params:, :] = 10*np.log10(data[:, added_params:, :]) # dB scale
        dBlim = 55  # define dynamic range
        data[data < -dBlim] = -dBlim  # set spectrum <-55 to -55
        data[:, added_params:, :] = data[:, added_params:, :]/dBlim + 1

    elif normalization == 'manual':
        m_max = 10369993.175721595 # SC spectral domain
        print('max:', m_max)
        data = data/m_max


    # the number of grid points
    i_x = data.shape[1] - added_params

    # Make the time series
    num_evo = train_evo + test_evo
    evo_size = steps - 1
    num_samples = np.round(num_evo*evo_size).astype(int)
    X_data_series = np.zeros((num_samples, window_size, i_x + added_params))
    Y_data_series = np.zeros((num_samples, i_x))

    for evo in range(num_evo):
        evo_data = np.transpose(data[evo, :, :])

        # tile the beginning of the evolution with 'window_size' input profiles
        temp1 = evo_data[0, :]
        temp2 = np.tile(temp1, (window_size - 1,1))
        evo_data = np.vstack((temp2, evo_data))

        for step in range(evo_size):
            input_data = evo_data[step:step + window_size, :]
            # remove additional parameters from output
            output_data = evo_data[step + window_size, added_params:]
            series_idx = evo*evo_size + step
            X_data_series[series_idx, :, :] = input_data
            Y_data_series[series_idx, :] = output_data

    X_train = X_data_series[:num_samples - test_evo*evo_size]
    X_test = X_data_series[num_samples - test_evo*evo_size:]
    Y_train = Y_data_series[:num_samples - test_evo*evo_size]
    Y_test = Y_data_series[num_samples - test_evo*evo_size:]

    return i_x, X_train, X_test, Y_train, Y_test
