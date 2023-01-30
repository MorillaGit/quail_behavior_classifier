import pandas as pd
import numpy as np
from brainflow.data_filter import (
    DataFilter,
    #     FilterTypes,
    #     AggOperations,
    DetrendOperations,
)

# import logging
# import matplotlib.pyplot as plt
import scipy.interpolate
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
)
import random


def conver_datframe_to_numpy(
    data_dataframe: pd.DataFrame,  # TODO: conver to convert, datframe to dataframe
    columns: list = None,
    debug: bool = True,  # TODO: unused
) -> np.ndarray:

    """ "  This function convert dataframe columns to numpy array

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
def detrend_signal(signal: np.ndarray, type_detrend: str = "CONSTANT") -> np.ndarray:

    """This function detrend the signal, applies the `detrend` function to a set of data.
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


def absolute_value(signal: np.ndarray) -> np.ndarray:

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
def envelope_aux(data: np.ndarray, distance: int) -> np.ndarray:

    """This function is a auxiliary, receives a np.array and returns a np.array, applied envelope to the data for graph


    :param data: the array to apply interpolation with envelope
    :type data: np.ndarray
    :param distance: is the sampling rate of the data, or distance between peaks
    :type distance: np.ndarray
    :return: the numpy array with the envelope
    :rtype: np.ndarray
    """

    datito = data
    # interpolation = distance

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
        u_peaks, _ = scipy.signal.find_peaks(u_y, distance=distance)

        # use peaks and peak values to make envelope
        u_x = u_peaks
        u_y = sig[u_peaks]

        # add start and end of signal to allow proper indexing
        end = len(sig)
        u_x = np.concatenate((u_x, [0, end]))
        u_y = np.concatenate((u_y, [0, 0]))

        # create envelope functions
        inter = scipy.interpolate.interp1d(u_x, u_y, kind="cubic")

        arrray.append(np.array(inter(x)))

        # convert to numpy array
    arrray = np.array(arrray)

    return arrray


def envelope(data: np.ndarray, distance: int) -> np.ndarray:
    """This function receives a np.array and returns a np.array, applied envelope with criteria max peak to the data.


    :param data: the array to apply interpolation with envelope
    :type data: np.ndarray
    :param distance: is the sampling rate of the data, or distance between peaks
    :type distance: np.ndarray
    :return: the numpy array with the envelope
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
        u_peaks, _ = scipy.signal.find_peaks(u_y, distance=distance)

        # use peaks and peak values to interpolate
        u_x = u_peaks
        u_y = u_y[u_peaks]

        # add start and end of signal to allow proper indexing
        end = len(sig)
        u_x = np.concatenate((u_x, [0, end]))
        u_y = np.concatenate((u_y, [0, 0]))

        # create envelope functions
        inter = scipy.interpolate.interp1d(
            u_x, u_y, kind="cubic"
        )  # TODO convert kind to parameter

        data_enineering.append(inter(x))

        # data_enineering = np.array(data_enineering)

    return np.array(data_enineering)


def normalize_data(
    data_df: pd.DataFrame,
    # data_np: np.ndarray = None,
    columns_scale: list = None,
    columns_no_scale: list = None,
    is_dataframe: bool = True,
    type_normalization: str = "MinMaxScaler",
) -> pd.DataFrame:

    """This function receives a pd.DataFrame or np.array and returns a pd.DataFrame normalized. Is possible to select the columns to normalize.


    :param data_df: the dataframe to apply normalization
    :type data_df: pd.DataFrame
    :param data_np: the array to apply normalization
    :type data_np: np.ndarray optional
    :param columns_scale: the columns to apply normalization
    :type columns_scale: list optional
    :param columns_no_scale: the columns to not apply normalization
    :type columns_no_scale: list optional
    :param is_dataframe: is a flag to know if the data is a dataframe or a numpy array
    :type is_dataframe: bool optional
    :param type_normalization: the type of normalization to apply, for more information see https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    :type type_normalization: str optional, default "MinMaxScaler"
    :return: pd.DataFrame with the data normalized
    :rtype: pd.DataFrame
    """
    if is_dataframe:
        # get data
        if columns_scale is None:
            # use all columns
            columns_scale = data_df.columns
        data = data_df[columns_scale].values
        # normalize data

    if type_normalization == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif type_normalization == "StandardScaler":
        scaler = StandardScaler()
    elif type_normalization == "RobustScaler":
        scaler = RobustScaler()
    elif type_normalization == "Normalizer":
        scaler = Normalizer()
    elif type_normalization == "MaxAbsScaler":
        scaler = MaxAbsScaler()
    elif type_normalization == "QuantileTransformer":
        scaler = QuantileTransformer()
    elif type_normalization == "PowerTransformer":
        scaler = PowerTransformer()
    else:
        scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=columns_scale)
    if columns_no_scale is not None:
        data = pd.concat([data_df[columns_no_scale], data], axis=1)
    return data


def split_windows(
    data: pd.DataFrame,
    exists_labels: bool = True,
    width_windows: int = 50,
    stride_windows: int = 50,
    balanced: bool = True,
    debug: bool = False,
) -> tuple:
    """This function receives a pd.DataFrame and returns a numpy array with this dimensions (n_windows, width_windows, n_channels) and the minimum number of windows for each label.


    :param data: the dataframe to apply split
    :type data: pd.DataFrame
    :param exists_labels: is a flag to know if the data has a column label
    :type exists_labels: bool optional, default True
    :param width_windows: the width of windows
    :type width_windows: int optional, default 50
    :param stride_windows: the stride of windows
    :type stride_windows: int optional, default 50
    :param balanced: is a flag to know if the data is balanced
    :type balanced: bool optional, default True
    :param debug: is a flag to know if the function is in debug mode
    :type debug: bool optional, default False
    :return: tuple with a list of pd.DataFrame and the minimum number of classes unbalanced
    :rtype: tuple
    """

    df_list = []
    if stride_windows is None:
        stride_windows = width_windows
    for i in range(0, len(data), stride_windows):
        df_list.append(data.iloc[i : i + width_windows])

    if exists_labels:
        df_label_windows = []
        for i in range(len(df_list)):
            # take a max value of segment TODO parametrize this criterion
            label = df_list[i]["label"].max()
            df_label_windows.append(label)

        for i in range(len(df_list)):
            df_list[i] = df_list[i].drop("label", axis=1)
            df_list[i] = df_list[i].values

        df_list = list(zip(df_list, df_label_windows))

        # count a number of windows with different labels
        count_0 = 0
        count_1 = 0
        count_2 = 0
        for i in range(len(df_list)):
            if df_list[i][1] == 0:
                count_0 += 1
            elif df_list[i][1] == 1:
                count_1 += 1
            elif df_list[i][1] == 2:
                count_2 += 1
        if debug:
            print(
                "--------------------Numbers of windows per class---------------------"
            )
            print(
                "Number of windows with normal behavior : ",
                count_0,
                "\nNumber of windows with reproductive event :  ",
                count_1,
                "\nNumber of windows with event of interest :  ",
                count_2,
                "\n\n",
            )

        # find a min number of windows with different labels
        min_count = min(count_0, count_1, count_2)
        if debug:
            print("The minimum number of windows per class is : ", min_count, "\n\n")

        # selecta a equal number of windows with different labels
        df_list_0 = []
        df_list_1 = []
        df_list_2 = []

        for i in range(len(df_list)):
            if df_list[i][1] == 0:
                df_list_0.append(df_list[i][0])
            elif df_list[i][1] == 1:
                df_list_1.append(df_list[i][0])
            elif df_list[i][1] == 2:
                df_list_2.append(df_list[i][0])

        # if debug:
        #     print("----------Numbers of windows per class-----------")
        #     print(  "Number of windows with normal behavior : ",  len(df_list_0),
        #             "\nNumber of windows with reproductive event :  ", len(df_list_1),
        #             "\nNumber of windows with event of interest :  ", len(df_list_2), "\n\n")

        df_class_0 = random.sample(df_list_0, min_count)
        df_class_1 = random.sample(df_list_1, min_count)
        df_class_2 = random.sample(df_list_2, min_count)

        if debug:
            print("----Numbers of windows per class before balanced data-----")
            print(
                "Number of windows with normal behavior : ",
                len(df_class_0),
                "\nNumber of windows with reproductive event :  ",
                len(df_class_1),
                "\nNumber of windows with event of interest :  ",
                len(df_class_2),
                "\n\n",
            )

        arr_0 = np.array(df_class_0, dtype=object)
        arr_1 = np.array(df_class_1, dtype=object)
        arr_2 = np.array(df_class_2, dtype=object)

        if debug:
            if exists_labels:
                print("-----------before balanced----------------")
                print(
                    "Number of windows with normal behavior : ",
                    len(arr_0),
                    "\nNumber of windows with reproductive event :  ",
                    len(arr_1),
                    "\nNumber of windows with event of interest :  ",
                    len(arr_2),
                    "\n\n",
                )

                print(
                    "\n",
                    "Number of windows with normal behavior : ",
                    len(arr_0),
                    "\n",
                    "The shape of each window is : ",
                    arr_0[0].shape,
                    "\n",
                    "The shape of the array is : ",
                    arr_0.shape,
                    "\n",
                    "The type of each window is : ",
                    type(arr_0[0]),
                    "\n\n",
                )

                print(
                    "\n",
                    "Number of windows with reproductive event : ",
                    len(arr_1),
                    "\n",
                    "The shape of each window is : ",
                    arr_1[0].shape,
                    "\n",
                    "The shape of the array is : ",
                    arr_1.shape,
                    "\n",
                    "The type of each window is : ",
                    type(arr_1[0]),
                    "\n\n",
                )

                print(
                    "\n",
                    "Number of windows with event of interest : ",
                    len(arr_2),
                    "\n",
                    "The shape of each window is : ",
                    arr_2[0].shape,
                    "\n",
                    "The shape of the array is : ",
                    arr_2.shape,
                    "\n",
                    "The type of each window is : ",
                    type(arr_2[0]),
                    "\n\n",
                )

        if balanced:
            return arr_0, arr_1, arr_2, min_count
        else:
            return df_list
    # TODO add mode no label
    # if exists_labels:
    #     arr_0 = arr_0[:, :, :-1]
    #     arr_1 = arr_1[:, :, :-1]
    #     arr_2 = arr_2[:, :, :-1]
    #     if debug:
    #         print("----------------whit label--------")
    #         print(  "\n","Number of windows with normal behavior : ",       len(arr_0),
    #                 "\nNumber of windows with reproductive event :  ",      len(arr_1),
    #                 "\nNumber of windows with event of interest :  ",      len(arr_2))

    #         print(      "\n","Number of windows with normal behavior : ", len(arr_0),
    #                     "\n","The shape of each window is : ", arr_0[0].shape,
    #                     "\n","The shape of the array is : ", arr_0.shape,
    #                     "\n", "The type of each window is : ", type(arr_0[0]))

    #         print(      "\n","Number of windows with reproductive event : ", len(arr_1),
    #                     "\n","The shape of each window is : ", arr_1[0].shape,
    #                     "\n","The shape of the array is : ", arr_1.shape,
    #                     "\n", "The type of each window is : ", type(arr_1[0]))

    #         print(      "\n","Number of windows with event of interest : ", len(arr_2),
    #                     "\n","The shape of each window is : ", arr_2[0].shape,
    #                     "\n","The shape of the array is : ", arr_2.shape,
    #                     "\n", "The type of each window is : ", type(arr_2[0]))
    # else:
    #     # arr_no_labeled = arr_no_labeled[:, :, :-1]
    #     if debug:
    #         print("----------------whiteout label--------")

    #         print(      "\n","Number of windows with normal behavior : ", len(arr_no_labeled),
    #                     "\n","The shape of each window is : ", arr_no_labeled[0].shape,
    #                     "\n","The shape of the array is : ", arr_no_labeled.shape,
    #                     "\n", "The type of each window is : ", type(arr_no_labeled[0]))

    #     return arr_no_labeled


def split_windows_2_class(
    data: pd.DataFrame,
    exists_labels: bool = True,
    width_windows: int = 50,
    stride_windows: int = 50,
    debug: bool = False,  # TODO fix error print
) -> tuple:
    """This function receives a pd.DataFrame and returns a numpy array with this dimensions (n_windows, width_windows, n_channels) and the minimum number of windows for each label.


    :param data: the dataframe to apply split
    :type data: pd.DataFrame
    :param exists_labels: is a flag to know if the data has a column label
    :type exists_labels: bool optional, default True
    :param width_windows: the width of windows
    :type width_windows: int optional, default 50
    :param stride_windows: the stride of windows
    :type stride_windows: int optional, default 50
    :param debug: is a flag to know if the function is in debug mode
    :type debug: bool optional, default False
    :return: tuple with a list of pd.DataFrame and the minimum number of classes unbalanced
    :rtype: tuple
    """

    df_list = []
    # convert dataframe in list
    if stride_windows is None:
        stride_windows = width_windows
    for i in range(0, len(data), stride_windows):
        df_list.append(data.iloc[i : i + width_windows])

    if exists_labels:
        df_label_windows = []
        for i in range(len(df_list)):
            # take a max value of segment TODO parametrize this criterion
            label = df_list[i]["label"].max()
            df_label_windows.append(label)

        data = data.drop(columns=["label"])

        # window = window.copy()
        # window['label'] = label

        label = int(label)
        for window, label in zip(df_list, df_label_windows):
            window.loc[:, "label"] = label  # -------error

        for window in range(len(df_list)):
            df_list[window] = pd.DataFrame(df_list[window])

        # count a number of windows with different labels
        count_0 = 0
        count_1 = 0
        # count_2 = 0
        for i in df_label_windows:
            if i == 0:
                count_0 += 1
            if i == 1:
                count_1 += 1
            # if i == 2:
            #     count_2 += 1
        if debug:
            print(
                "--------------------Numbers of windows per class---------------------"
            )
            print(
                "Number of windows with normal behavior : ",
                count_0,
                "\nNumber of windows with reproductive event :  ",
                count_1,
            )
            # "\nNumber of windows with event of interest :  ",   count_2)

        # find a min number of windows with different labels
        min_count = min(count_0, count_1)

        # selecta a equal number of windows with different labels
        df_list_0 = []
        df_list_1 = []
        df_list_2 = []

        for i in range(len(df_list)):
            if df_label_windows[i] == 0:
                df_list_0.append(df_list[i])
            if df_label_windows[i] == 1:
                df_list_1.append(df_list[i])
            # if df_label_windows[i] == 2:
            #     df_list_2.append(df_list[i])

        # if debug:
        #     print("----------Numbers of windows per class-----------")
        #     print(  "Number of windows with normal behavior : ",  len(df_list_0),
        #             "\nNumber of windows with reproductive event :  ", len(df_list_1),
        #             "\nNumber of windows with event of interest :  ", len(df_list_2))

        df_class_0 = random.sample(df_list_0, min_count)
        df_class_1 = random.sample(df_list_1, min_count)
        # df_class_2 = random.sample(df_list_2, min_count)

        if debug:
            print("----Numbers of windows per class before balanced data-----")
            print(
                "Number of windows with normal behavior : ",
                len(df_class_0),
                "\nNumber of windows with reproductive event :  ",
                len(df_class_1),
            )
            # "\nNumber of windows with event of interest :  ", len(df_class_2))

    # transform list of dataframes to numpy array whit shape = (len(df_class), 50, features)
    def list_to_array(lista: list) -> np.ndarray:
        array = []
        for i in range(len(lista)):
            array.append(lista[i].values)
        return np.array(array)

    if exists_labels:
        arr_0 = list_to_array(df_class_0)
        arr_1 = list_to_array(df_class_1)
        # arr_2 = list_to_array(df_class_2)

    arr_no_labeled = list_to_array(lista=df_list)

    if debug:
        if exists_labels:
            print("-----------before balanced----------------")
            print(
                "\n",
                "Number of windows with normal behavior : ",
                len(arr_0),
                "\nNumber of windows with reproductive event :  ",
                len(arr_1),
            )
            # "\nNumber of windows with event of interest :  ",   len(arr_2))

            print(
                "\n",
                "Number of windows with normal behavior : ",
                len(arr_0),
                "\n",
                "The shape of each window is : ",
                arr_0[0].shape,
                "\n",
                "The shape of the array is : ",
                arr_0.shape,
                "\n",
                "The type of each window is : ",
                type(arr_0[0]),
            )

            print(
                "\n",
                "Number of windows with reproductive event : ",
                len(arr_1),
                "\n",
                "The shape of each window is : ",
                arr_1[0].shape,
                "\n",
                "The shape of the array is : ",
                arr_1.shape,
                "\n",
                "The type of each window is : ",
                type(arr_1[0]),
            )

            # print(      "\n","Number of windows with event of interest : ", len(arr_2),
            #             "\n","The shape of each window is : ", arr_2[0].shape,
            #             "\n","The shape of the array is : ", arr_2.shape,
            #             "\n", "The type of each window is : ", type(arr_2[0]))
    if exists_labels:
        arr_0 = arr_0[:, :, :-1]
        arr_1 = arr_1[:, :, :-1]
        arr_2 = arr_2[:, :, :-1]
        if debug:
            print("----------------whit label--------")
            print(
                "\n",
                "Number of windows with normal behavior : ",
                len(arr_0),
                "\nNumber of windows with reproductive event :  ",
                len(arr_1),
                "\nNumber of windows with event of interest :  ",
                len(arr_2),
            )

            print(
                "\n",
                "Number of windows with normal behavior : ",
                len(arr_0),
                "\n",
                "The shape of each window is : ",
                arr_0[0].shape,
                "\n",
                "The shape of the array is : ",
                arr_0.shape,
                "\n",
                "The type of each window is : ",
                type(arr_0[0]),
            )

            print(
                "\n",
                "Number of windows with reproductive event : ",
                len(arr_1),
                "\n",
                "The shape of each window is : ",
                arr_1[0].shape,
                "\n",
                "The shape of the array is : ",
                arr_1.shape,
                "\n",
                "The type of each window is : ",
                type(arr_1[0]),
            )

            # print(      "\n","Number of windows with event of interest : ", len(arr_2),
            #             "\n","The shape of each window is : ", arr_2[0].shape,
            #             "\n","The shape of the array is : ", arr_2.shape,
            #             "\n", "The type of each window is : ", type(arr_2[0]))
        return arr_0, arr_1, min_count
    else:
        # arr_no_labeled = arr_no_labeled[:, :, :-1]
        if debug:
            print("----------------whiteout label--------")

            print(
                "\n",
                "Number of windows with normal behavior : ",
                len(arr_no_labeled),
                "\n",
                "The shape of each window is : ",
                arr_no_labeled[0].shape,
                "\n",
                "The shape of the array is : ",
                arr_no_labeled.shape,
                "\n",
                "The type of each window is : ",
                type(arr_no_labeled[0]),
            )

        return arr_no_labeled
