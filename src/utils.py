import pandas as pd
import numpy as np
import logging

# add description to module

def load_data(  
                path_data: str = "data/",
                name_csv_features: str = "features.csv",
                name_csv_labels: str = None,  # TODO None to "", is not for !=
                delay: int = 150, 
                labeled_data_step: int = 600000,
                )-> pd.DataFrame:
    """ Load data from csv files and convert to pandas DataFrame \n
   Loads the data and labels from the specified .csv files and returns them as a tuple. 
   The data is loaded from the file specified by `name_csv_features` and the labels are loaded 
   from the file specified by `name_csv_labels`. The `delay` parameter specifies the difference 
   between the data and the labels, and the `labeled_data_step` parameter specifies the last 
   labeled data point in the dataset. 

    :param path_data: the path where the .csv files are stored
    :type path_data: str
    :param name_csv_features: the name of the .csv file that contains the data
    :type name_csv_features: str
    :param name_csv_labels: the name of the .csv file that contains the labels
    :type name_csv_labels: str
    :param delay: the difference between the data and the labels
    :type delay: int
    :param labeled_data_step: the last labeled data point in the dataset
    :type labeled_data_step: int
    :return: a tuple with the loaded data and labels
    :rtype:  pd.DataFrame
    """

    inputs = pd.read_csv(path_data + name_csv_features, low_memory=False)

    # if name_csv_labels is different to None, load the labels else dont use labels
    if name_csv_labels is not None:
        labels = pd.read_csv(path_data + name_csv_labels, low_memory=False)

    # crop data for synchronization
    inputs = pd.DataFrame(inputs[delay:])

    inputs.reset_index(drop=True, inplace=True)

    if name_csv_labels is not None:
        df = pd.concat([inputs, labels], axis=1)
        # the label is not float. it's a category convert to int
        df['label'] = df['label'].astype('int')
    else:
        df = inputs

    if labeled_data_step is not None:
        # use the first labeled_data_step samples because just [0:600000] are labeled
        df = df[:labeled_data_step]

    return df

def arr_to_dataframe(   data_to_add: np.ndarray,
                        data_base: pd.DataFrame,
                        names_new_columns: list,
                        debug : bool = False,
                        ) -> pd.DataFrame:
    """This function convert a numpy array to a pandas DataFrame and add to the data_base

    :param data_to_add: data in format array TODO complete
    :type data_to_add: np.ndarray
    :param data_base: data in format DataFrame, data raw
    :type data_base: pd.DataFrame
    :param names_new_columns: names of the new columns, the number of columns must be equal to the number of columns of data_to_add
    :type names_new_columns: list
    :param debug: this parameter is used for debug the function, printing the name of columns generated, defaults to False
    :type debug: bool, optional
    :return: a DataFrame with the data_base and the data_to_add
    :rtype: pd.DataFrame
    """

    data_to_add = pd.DataFrame(data_to_add.T, columns=names_new_columns)

    data_eng = pd.concat([data_base, data_to_add], axis=1)

    cols = list(data_eng.columns)

    # if exists labels columns, move to the end
    if "label" in data_eng.columns:
        # move labels to the end TODO refactor this shit
        cols = cols[-1:] + cols[:-1]
        cols = cols[-1:] + cols[:-1]
        cols = cols[-1:] + cols[:-1]

    data_eng = data_eng[cols]

    if debug:
        print("The columns are ", data_eng.columns)
    
    return data_eng

def create_labels(
                dim_0: int, 
                debug: bool = False
                ) -> tuple:
    """This function create labels for the data, is necessary introduce the dimension of the data

    :param dim_0: the dimension of the data minimum represent the number of samples
    :type dim_0: int
    :param debug: this parameter is used for debug the function, printing the shape of data generated, defaults to False
    :type debug: bool, optional
    :return: a tuple with the labels
    :rtype: tuple
    """
    # create labels
    y_train_0 = np.zeros(dim_0)
    y_train_1 = np.ones(dim_0)
    y_train_2 = np.full(dim_0, 2)

    y_test_0 = np.zeros(dim_0)
    y_test_1 = np.ones(dim_0)
    y_test_2 = np.full(dim_0, 2)
    
    if debug:
        print("--------------the shape if Train per class------------------") # TODO refactor
        print(  "\n","y_train_0 shape : ", y_train_0.shape,
                "\n","y_train_1 shape : ", y_train_1.shape,
                "\n","y_train_2 shape : ", y_train_2.shape)

        print("--------------the shape if test per class------------------")
        print(  "\n","y_test_0 shape : ", y_test_0.shape,
                "\n","y_test_1 shape : ", y_test_1.shape,
                "\n","y_test_2 shape : ", y_test_2.shape)
                
    return y_train_0, y_train_1, y_train_2


