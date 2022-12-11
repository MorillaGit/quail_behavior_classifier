import pandas as pd
import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations,DetrendOperations
import logging
import matplotlib.pyplot as plt
import scipy.interpolate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random


def conver_datframe_to_numpy(   data_dataframe: pd.DataFrame, # TODO: conver to convert, datframe to dataframe
                                columns: list = None,
                                debug : bool = True, # TODO: unused
                                ) -> np.ndarray:

    """"  This function convert dataframe columns to numpy array

    :param data_dataframe: the dataframe to convert
    :type data_dataframe: pd.DataFrame
    :param columns: columns to convert
    :type columns: list
    :param debug: this is a debug flag, to see a dimension of the data
    :type debug: bool
    :return: the numpy array
    :rtype:  np.ndarray
    """
    if columns is not None:
        if debug:
            print("The columns are ", data_dataframe[columns].shape)
        return data_dataframe[columns].to_numpy()

    if debug:
            print("The columns are ", data_dataframe.shape)

    return data_dataframe.to_numpy()

# TODO: the detrand not is necessary, because the data is already detrended
def detrend_signal( signal: np.ndarray,
                    type_detrend : str = "CONSTANT"
                    ) -> np.ndarray:

    """ This function detrend the signal, applies the `detrend` function to a set of data. 
    This function removes the linear trend from the data, leaving only the random 
    fluctuations. The detrended data is returned. The type of detrend is specified by
    the `type_detrend` parameter.
    
    :param signal: the array to detrend
    :type signal: np.ndarray
    :param type_detrend: the type of detrend, by default "CONSTANT"
    :type type_detrend: str, optional
    :return: the numpy array with the detrended signal
    :rtype: np.ndarray
    """
    for i in range(signal.shape[1]):
        if type_detrend == "LINEAR":
            DataFilter.detrend(signal.T[i], DetrendOperations.LINEAR.value)
        elif type_detrend == "CONSTANT":
            DataFilter.detrend(signal.T[i], DetrendOperations.CONSTANT.value)
        else:
            DataFilter.detrend(signal.T[i], DetrendOperations.CONSTANT.value)
    return signal


def absolute_value( signal: np.ndarray
                    ) -> np.ndarray:

    """This function receives a numpy array and returns the absolute value of the array
    
    :param signal: the array to apply absolute value
    :type signal: np.ndarray
    :return: the numpy array with the absolute value
    :rtype: np.ndarray
    """
    signal = np.absolute(signal)

    return signal

# this function is made in FACU TODO Refactor
# this function calculate envelope of the data, receives a np.array,and integer and returns a np.array
def envelope_aux(data: np.ndarray, sampling_rate: int) -> np.ndarray:

    """ This function calculate envelope of the data
    
    Parameters
    ----------
    data : np.ndarray
        data to calculate envelope
    sampling_rate : int
        sampling rate

    Returns
    -------
    np.ndarray
        data with envelope
    # """

    datito = data
    interpolation = sampling_rate

    x = np.arange(len(datito))
    inter = []
    arrray = []

    for channel in range(datito.shape[1]):
    # print(channel)
        sig = datito.T[channel]
        # inter[channel] = envelope(sig, interpolation)

        u_x = np.where(sig > 0)[0]
        u_y = sig.copy()

        # find upper peaks
        u_peaks, _ = scipy.signal.find_peaks(u_y, distance=interpolation)

        # use peaks and peak values to make envelope
        u_x = u_peaks
        u_y = sig[u_peaks]

        # add start and end of signal to allow proper indexing
        end = len(sig)
        u_x = np.concatenate((u_x, [0, end]))
        u_y = np.concatenate((u_y, [0, 0]))

        # create envelope functions
        inter = scipy.interpolate.interp1d(u_x, u_y,kind='cubic')
        
        # print(channel)
        arrray.append(np.array(inter(x)))

        # convert to numpy array
    arrray = np.array(arrray)

    return arrray

def  feature_engineering(   data: np.ndarray,  # TODO renamed Envelope
                            interpolation: int
                            ) -> np.ndarray:
    """ This function receives a np.array and returns a np.array, applied envelope to the data
    
    
    :param signal: the array to apply absolute value
    :type signal: np.ndarray
    :return: the numpy array with the absolute value
    :rtype: np.ndarray
    """
    x = np.arange(len(data))
    inter = []
    data_enineering = []

    for channel in range(data.shape[1]):
        # calculate the envelope
        sig = data.T[channel]
        
        u_x = np.where(sig > 0)[0]
        u_y = sig.copy()

        # find upper peaks
        u_peaks, _ = scipy.signal.find_peaks(u_y, distance=interpolation)

        # use peaks and peak values to interpolate
        u_x = u_peaks
        u_y = u_y[u_peaks]

        # add start and end of signal to allow proper indexing
        end = len(sig)
        u_x = np.concatenate((u_x, [0, end]))
        u_y = np.concatenate((u_y, [0, 0]))

        # create envelope functions
        inter = scipy.interpolate.interp1d(u_x, u_y,kind='cubic')  # TODO convert kind to parameter
        
        data_enineering.append(inter(x))

    return data_enineering


def normalize_data(
                data_df: pd.DataFrame, 
                data_np: np.ndarray = None,     
                columns_scale: list = None,
                columns_labels: list = None,    
                is_dataframe: bool = True, 
                type_normalization : callable = MinMaxScaler  # TODO  set a options type str
                ) -> pd.DataFrame:
                # TODO traking y debug arguments

    """ This function receives a dataframe or a np.array and returns a dataframe or a np.array, applied normalization
    
    Parameters
    ----------
    data_df : pd.DataFrame
        dataframe to normalize
    data_np : np.ndarray, optional
        np.array to normalize, by default None
    labels : list
        list of labels
    is_dataframe : bool, optional
        boolean to know if the data is a dataframe or a np.array, by default True
    type_norm : str, optional
        type of normalization, by default MinMaxScaler

    Returns
    -------
    pd.DataFrame
        dataframe with normalization
    """
    if is_dataframe:
        # get data
        if columns_scale is None:
            # use all columns
            columns_scale = data_df.columns
        data = data_df[columns_scale].values
        # normalize data
    scaler = type_normalization()
    data = scaler.fit_transform(data)
    # convert to dataframe
    data = pd.DataFrame(data, columns=columns_scale)
    # add to dataframe columns labels
    if columns_labels is not None:
        data = pd.concat([data_df[columns_labels], data], axis=1)
        # move column labels to the end TODO refactor this shit
        cols = list(data.columns)
        cols = cols[-1:] + cols[:-1]
        cols = cols[-1:] + cols[:-1]
        cols = cols[-1:] + cols[:-1]
        cols = cols[-1:] + cols[:-1]
        cols = cols[-1:] + cols[:-1]
        cols = cols[-1:] + cols[:-1]
        data = data[cols]

    return data


# TODO solve this problem
# SettingWithCopyWarning: 
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead

# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  # add to dataframe columns labels
def preposses_balanced( # TODO rename to preposses_balanced
                data : pd.DataFrame,
                exists_labels : bool = True,
                width_windows : int = 50, 
                stride_windows : int = 50, 
                debug : bool = False
                ) -> tuple:
    """ This function receives a dataframe and returns a dataframe, applied windows
    
    Parameters
    ----------
    data : pd.DataFrame
        dataframe to apply windows
    exists_labels : bool, optional
        boolean to know if the dataframe has labels, by default True
    width_windows : int
        width of windows
    stride_windows : int, optional
        stride of windows, by default None

    Returns
    -------
    pd.DataFrame
        dataframe with windows
    """
    # this part take a dataframe and return a list of dataframes
    df_list = []
    # convert dataframe in list
    if stride_windows is None:
        stride_windows = width_windows    
    for i in range(0, len(data), stride_windows):
        df_list.append(data.iloc[i:i+width_windows])

    if exists_labels:
    # this part take a list of dataframes and transform a label column to columns of max in this windows
        df_label_windows = []
        for i in range(len(df_list)):
            # take a max value of segment
            label = df_list[i]['label'].max()
            # create a new dataframe with label
            df_label_windows.append(label)

    # drop a column label of data
        data = data.drop(columns=['label'])
        # this part add a column with label to dataframe
        for window, label in zip(df_list, df_label_windows):
            # window['label'] = label
            window.loc[:, 'label'] = label #--------------------------------

    
        # conver listof windows to dataframe
        for window in range(len(df_list)):
            df_list[window] = pd.DataFrame(df_list[window])

        # count a number of windows with different labels
        count_0 = 0
        count_1 = 0
        count_2 = 0
        for i in df_label_windows:
            if i == 0:
                count_0 += 1
            if i == 1:
                count_1 += 1
            if i == 2:
                count_2 += 1  
        if debug:
            print("-----------------1----------------------")
            print("Number of windows with normal behaviour : ", count_0,
                "\nNumber of windows with reproductive event :  ", count_1,
                "\nNnumber of windows with event of interest :  ", count_2)

        # find a min number of windows with different labels
        min_count = min(count_0, count_1, count_2)

        # selecta a equal number of windows with different labels
        df_list_0 = []
        df_list_1 = []
        df_list_2 = []

        for i in range(len(df_list)):
            if df_label_windows[i] == 0:
                df_list_0.append(df_list[i])
            if df_label_windows[i] == 1:
                df_list_1.append(df_list[i])
            if df_label_windows[i] == 2:
                df_list_2.append(df_list[i])

        if debug:
            print("-----------------2----------------------")
            print(  "Number of windows with normal behaviour : ",  len(df_list_0),
                    "\nNumber of windows with reproductive event :  ", len(df_list_1),
                    "\nNnumber of windows with event of interest :  ", len(df_list_2))
                
        # df_class_

        df_class_0 = random.sample(df_list_0, min_count)
        df_class_1 = random.sample(df_list_1, min_count)
        df_class_2 = random.sample(df_list_2, min_count)


        if debug:
            print("-----------------3----------------------")
            print(  "Number of windows with normal behaviour : ",  len(df_class_0),
                    "\nNumber of windows with reproductive event :  ", len(df_class_1),
                    "\nNnumber of windows with event of interest :  ", len(df_class_2))

    # transform list of dataframes to numpy array whit shape = (len(df_class), 50, features)
    def list_to_array(lista: list) -> np.ndarray:
        array = []
        for i in range(len(lista)):
            array.append(lista[i].values)
        return np.array(array)

    if exists_labels:
        arr_0 = list_to_array(df_class_0)
        arr_1 = list_to_array(df_class_1)
        arr_2 = list_to_array(df_class_2)

    arr_no_labeled = list_to_array(lista=df_list)


    if debug:
        if exists_labels:
            print("-----------------4----------------------")
            print(  "\n","Number of windows with normal behavior : ",       len(arr_0),
                    "\nNumber of windows with reproductive event :  ",      len(arr_1),
                    "\nNnumber of windows with event of interest :  ",      len(arr_2))

            print(      "\n","Number of windows with normal behavior : ", len(arr_0),        
                        "\n","The shape of each window is : ", arr_0[0].shape,
                        "\n","The shape of the array is : ", arr_0.shape,
                        "\n", "The type of each window is : ", type(arr_0[0]))

            print(      "\n","Number of windows with reproductive event : ", len(arr_1),        
                        "\n","The shape of each window is : ", arr_1[0].shape,
                        "\n","The shape of the array is : ", arr_1.shape,
                        "\n", "The type of each window is : ", type(arr_1[0]))

            print(      "\n","Number of windows with event of interest : ", len(arr_2),        
                        "\n","The shape of each window is : ", arr_2[0].shape,
                        "\n","The shape of the array is : ", arr_2.shape,
                        "\n", "The type of each window is : ", type(arr_2[0]))
    if exists_labels:
# Eliminate the last column of each window
        arr_0 = arr_0[:, :, :-1]
        arr_1 = arr_1[:, :, :-1]
        arr_2 = arr_2[:, :, :-1]
        if debug:
            print("-----------------5--------whiteout label--------")
            print(  "\n","Number of windows with normal behavior : ",       len(arr_0),
                    "\nNumber of windows with reproductive event :  ",      len(arr_1),
                    "\nNnumber of windows with event of interest :  ",      len(arr_2))

            print(      "\n","Number of windows with normal behavior : ", len(arr_0),        
                        "\n","The shape of each window is : ", arr_0[0].shape,
                        "\n","The shape of the array is : ", arr_0.shape,
                        "\n", "The type of each window is : ", type(arr_0[0]))

            print(      "\n","Number of windows with reproductive event : ", len(arr_1),        
                        "\n","The shape of each window is : ", arr_1[0].shape,
                        "\n","The shape of the array is : ", arr_1.shape,
                        "\n", "The type of each window is : ", type(arr_1[0]))

            print(      "\n","Number of windows with event of interest : ", len(arr_2),        
                        "\n","The shape of each window is : ", arr_2[0].shape,
                        "\n","The shape of the array is : ", arr_2.shape,
                        "\n", "The type of each window is : ", type(arr_2[0]))
        return arr_0, arr_1, arr_2, min_count
    else:
        # arr_no_labeled = arr_no_labeled[:, :, :-1]
        if debug:
            print("-----------------5--------whiteout label--------")

            print(      "\n","Number of windows with normal behavior : ", len(arr_no_labeled),        
                        "\n","The shape of each window is : ", arr_no_labeled[0].shape,
                        "\n","The shape of the array is : ", arr_no_labeled.shape,
                        "\n", "The type of each window is : ", type(arr_no_labeled[0]))


        return arr_no_labeled

# TODO Diocito dice 
# # Charlas con el diocito 
# Para detectar anomalías en un dataset esparso, se pueden utilizar diferentes algoritmos de machine learning o deep learning, dependiendo de la naturaleza y las características del dataset en cuestión. Algunos ejemplos de algoritmos que se pueden utilizar en este caso son:

# Algoritmos de detección de anomalías basados en la densidad: estos algoritmos, como el clasificador de densidad de KNN (k-nearest neighbors), se basan en la idea de que las anomalías suelen ser valores atípicos dentro de un conjunto de datos, por lo que se pueden detectar como aquellos que tienen una densidad de vecinos cercanos muy baja en comparación con el resto de los datos.

# Algoritmos de detección de anomalías basados en el aprendizaje profundo: estos algoritmos, como las redes neuronales autoencoders o las redes generativas adversarias, se basan en la idea de que las anomalías suelen tener características distintivas en comparación con el resto de los datos, por lo que se pueden aprender a detectarlas mediante el entrenamiento de un modelo de deep learning con un conjunto de datos normal y anómalos.

# Algoritmos de detección de anomalías basados en el aprendizaje no supervisado: estos algoritmos, como el clustering de DBSCAN (density-based spatial clustering of applications with noise), se basan en la idea de que las anomalías suelen tener un comportamiento diferente al del resto de los datos, por lo que se pueden detectar como aquellos que no pertenecen a ningún grupo o cluster formado por el algoritmo de clustering.