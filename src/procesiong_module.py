import numpy as np
from sklearn.model_selection import train_test_split

def split_data(tuple_arr_class_X : tuple, tuple_arr_class_y : tuple, rate_split : float = 0.2, debug: bool = False, replicability : int = 42) -> tuple:
    """_summary_

    Args:
        tuple_arr_class_X (tuple): _description_
        tuple_arr_class_y (tuple): _description_
        rate_split (float, optional): _description_. Defaults to 0.2.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        tuple: _description_
    """
    # split data
    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(tuple_arr_class_X[0], tuple_arr_class_y[0], test_size=rate_split, random_state=replicability)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(tuple_arr_class_X[1], tuple_arr_class_y[1], test_size=rate_split, random_state=replicability)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(tuple_arr_class_X[2], tuple_arr_class_y[2], test_size=rate_split, random_state=replicability)

    if debug:
        print("------------X_train.shape-----------------")
        print(  "\n","X_train_0 shape : ", X_train_0.shape,
                "\n","X_train_1 shape : ", X_train_1.shape,
                "\n","X_train_2 shape : ", X_train_2.shape)

        print("------------X_test.shape---------------")
        print(  "\n","X_test_0 shape : ", X_test_0.shape,
                "\n","X_test_1 shape : ", X_test_1.shape,
                "\n","X_test_2 shape : ", X_test_2.shape)

        print("------------y_train.shape-----------------")
        print(  "\n","y_train_0 shape : ", y_train_0.shape,
                "\n","y_train_1 shape : ", y_train_1.shape,
                "\n","y_train_2 shape : ", y_train_2.shape)

        print("------------y_test.shape---------------")    
        print(  "\n","y_test_0 shape : ", y_test_0.shape,   
                "\n","y_test_1 shape : ", y_test_1.shape,
                "\n","y_test_2 shape : ", y_test_2.shape)

    # concatenate data
    X_train = np.concatenate((X_train_0, X_train_1, X_train_2), axis=0)
    X_test = np.concatenate((X_test_0, X_test_1, X_test_2), axis=0)
    y_train = np.concatenate((y_train_0, y_train_1, y_train_2), axis=0)
    y_test = np.concatenate((y_test_0, y_test_1, y_test_2), axis=0)

    if debug:
        print("------------X_train.shape-----------------")
        print(  "\n","X_train shape : ", X_train.shape)

        print("------------X_test.shape---------------")
        print(  "\n","X_test shape : ", X_test.shape)

        print("------------y_train.shape-----------------")
        print(  "\n","y_train shape : ", y_train.shape)

        print("------------y_test.shape---------------")    
        print(  "\n","y_test shape : ", y_test.shape)

    return X_train, X_test, y_train, y_test


def reshape_data(   X_train : np.ndarray, 
                        X_test : np.ndarray, 
                        y_train : np.ndarray, 
                        y_test : np.ndarray, 
                        debug: bool = False) -> tuple:
    """_summary_

    Args:
        X_train (np.ndarray): _description_
        X_test (np.ndarray): _description_
        y_train (np.ndarray): _description_
        y_test (np.ndarray): _description_
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        tuple: _description_
    """
    # reshape data
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    if debug:
        print("------------X_train.shape-----------------")
        print(  "\n","X_train shape : ", X_train.shape)

        print("------------X_test.shape---------------")
        print(  "\n","X_test shape : ", X_test.shape)

        print("------------y_train.shape-----------------")
        print(  "\n","y_train shape : ", y_train.shape)

        print("------------y_test.shape---------------")    
        print(  "\n","y_test shape : ", y_test.shape)

    return X_train, X_test, y_train, y_test


