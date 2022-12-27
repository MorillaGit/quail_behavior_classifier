import numpy as np
from sklearn.model_selection import train_test_split

def split_data(tuple_arr_class_X : tuple, tuple_arr_class_y : tuple, rate_split : float = 0.2, debug: bool = False, replicability : int = 42) -> tuple:
    """This function split the data in train and test

    :param tuple_arr_class_X: a tuple with the data in format array
    :type tuple_arr_class_X: tuple
    :param tuple_arr_class_y: a tuple with the labels in format array
    :type tuple_arr_class_y: tuple
    :param rate_split: the rate of split, defaults to 0.2
    :type rate_split: float, optional
    :param debug: this parameter is used for debug the function, printing the shape of data generated, defaults to False
    :type debug: bool, optional
    :param replicability: this parameter is used for replicability, defaults to 42
    :type replicability: int, optional
    :return: a tuple with the data split
    :rtype: tuple
    """
    # split data
    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(tuple_arr_class_X[0], tuple_arr_class_y[0], test_size=rate_split, random_state=replicability)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(tuple_arr_class_X[1], tuple_arr_class_y[1], test_size=rate_split, random_state=replicability)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(tuple_arr_class_X[2], tuple_arr_class_y[2], test_size=rate_split, random_state=replicability)

    if debug:
        print("------------X_train.shape-----------------")
        print(  "X_train_0 shape : ", X_train_0.shape,
                "\n","X_train_1 shape : ", X_train_1.shape,
                "\n","X_train_2 shape : ", X_train_2.shape,"\n")

        print("------------X_test.shape---------------")
        print(  "X_test_0 shape : ", X_test_0.shape,
                "\n","X_test_1 shape : ", X_test_1.shape,
                "\n","X_test_2 shape : ", X_test_2.shape,"\n")

        print("------------y_train.shape-----------------")
        print(  "y_train_0 shape : ", y_train_0.shape,
                "\n","y_train_1 shape : ", y_train_1.shape,
                "\n","y_train_2 shape : ", y_train_2.shape,"\n")

        print("------------y_test.shape---------------")    
        print(  "y_test_0 shape : ", y_test_0.shape,   
                "\n","y_test_1 shape : ", y_test_1.shape,
                "\n","y_test_2 shape : ", y_test_2.shape,"\n")

    # concatenate data
    X_train = np.concatenate((X_train_0, X_train_1, X_train_2), axis=0)
    X_test = np.concatenate((X_test_0, X_test_1, X_test_2), axis=0)
    y_train = np.concatenate((y_train_0, y_train_1, y_train_2), axis=0)
    y_test = np.concatenate((y_test_0, y_test_1, y_test_2), axis=0)

    if debug:
        print("------------X_train.shape-----------------")
        print("X_train shape : ", X_train.shape,"\n")

        print("------------X_test.shape-------------------")
        print("X_test shape : ", X_test.shape,"\n")

        print("------------y_train.shape-------------------")
        print("y_train shape : ", y_train.shape,"\n")

        print("------------y_test.shape--------------------") 
        print("y_test shape : ", y_test.shape,"\n")

    return X_train, X_test, y_train, y_test


def reshape_data(   X_train : np.ndarray, 
                        X_test : np.ndarray, 
                        y_train : np.ndarray, 
                        y_test : np.ndarray, 
                        debug: bool = False) -> tuple:
    """This function reshape the data for CNN model

    :param X_train: the data train
    :type X_train: np.ndarray
    :param X_test: the data test
    :type X_test: np.ndarray
    :param y_train: the labels train
    :type y_train: np.ndarray
    :param y_test: the labels test
    :type y_test: np.ndarray
    :param debug: this parameter is used for debug the function, printing the shape of data generated, defaults to False
    :type debug: bool, optional
    :return: a tuple with the data reshaped
    :rtype: tuple
    """
    # reshape data
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    if debug:
        print("------------X_train.shape-----------------")
        print(  "X_train shape : ", X_train.shape,"\n")

        print("------------X_test.shape-------------------")
        print(  "X_test shape : ", X_test.shape,"\n")

        print("------------y_train.shape-----------------")
        print(  "y_train shape : ", y_train.shape,"\n")

        print("------------y_test.shape-------------------")    
        print( "y_test shape : ", y_test.shape,"\n")

    return X_train, X_test, y_train, y_test


