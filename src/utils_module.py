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
                traking: bool = False 
                )-> pd.DataFrame:
    """_summary_

    :param path_data: path where is the data csv, defaults to "data/"
    :type path_data: str, optional
    :param name_csv_features: name of csv features, defaults to "features.csv"
    :type name_csv_features: str, optional
    :param name_csv_labels: path where is the data csv, defaults to None
    :type name_csv_labels: str, optional
    :param delay: _description_, defaults to 150
    :type delay: int, optional
    :param labeled_data_step: _description_, defaults to 600000
    :type labeled_data_step: int, optional
    :param traking: _description_, defaults to False
    :type traking: bool, optional
    :return: _description_
    :rtype: pd.DataFrame
    """

    inputs = pd.read_csv(path_data + name_csv_features, low_memory=False)

    # if name_csv_labels difers from None, load the labels else dont use labels
    if name_csv_labels is not None:
        labels = pd.read_csv(path_data + name_csv_labels, low_memory=False)

    # crop data for synchronization
    inputs = pd.DataFrame(inputs[delay:])


    # report delay in logging
    if traking:
        logging.info(f"Delay between features and labels: {delay} seconds")

    # reset index
    inputs.reset_index(drop=True, inplace=True)

    # df is composed by inputs and labels
    if name_csv_labels is not None:
        df = pd.concat([inputs, labels], axis=1)
        # the label is not float. it's a category
        # convert to int
        df['label'] = df['label'].astype('int')
    else:
        df = inputs

    if labeled_data_step is not None:
        # use the first labeled_data_step samples because just [0:600000] are labeled
        df = df[:labeled_data_step]

    # report process in logging TODO if tracking , correct traking to tracking
    logging.info(f"Data loaded from {path_data + name_csv_features}")

    return df

def arr_to_dataframe(   data_to_add: np.ndarray,
                        data_base: pd.DataFrame,
                        names_new_columns: list,
                        debug : bool = False, 
                        traking: bool = False
                        ) -> pd.DataFrame:
    """This function receives a numpy array and columns optionally and returns a dataframe

    Parameters
    ----------
    data_to_add : np.ndarray
        numpy array to convert
    names_new_columns : list, optional
        columns to convert, by default None
    debug : bool, optional
        debug mode, by default True
    traking : bool, optional
        traking mode, by default False

    Returns
    -------
    pd.DataFrame
        dataframe
    """

    # using logging # TODO find the better way to do this
    if traking:
        logging.info(msg="Converting numpy array to dataframe, the shape of the numpy array is: "
        + str(data_to_add.shape) +
        "name of the columns: " 
        + str(list_columns_add) +
        "name of function: arr_to_dataframe")

    # convert to dataframe
    data_to_add = pd.DataFrame(data_to_add.T, columns=names_new_columns)
    # add to dataframe
    data_eng = pd.concat([data_base, data_to_add], axis=1)

    cols = list(data_eng.columns)

    # if exists labels columns, move to the end
    if "label" in data_eng.columns:
        # move labels to the end TODO refactor this shit
        cols = cols[-1:] + cols[:-1]
        cols = cols[-1:] + cols[:-1]
        cols = cols[-1:] + cols[:-1]
        # cols = cols[-1:] + cols[:-1]


    data_eng = data_eng[cols]

    if debug:
        print("The columns are ", data_eng.columns)
    
    return data_eng

def create_labels(
                dim_0: int, 
                debug: bool = False
                ) -> tuple:
    """generate a vectior of length dim_0 

    Args:
        dim_0 (int): _description_
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        tuple: _description_
    """
    # create labels
    y_train_0 = np.zeros(dim_0)
    y_train_1 = np.ones(dim_0)
    y_train_2 = np.full(dim_0, 2)

    y_test_0 = np.zeros(dim_0)
    y_test_1 = np.ones(dim_0)
    y_test_2 = np.full(dim_0, 2)
    
    if debug:
        print("-----------------5----------------------") # TODO refactor
        print(  "\n","y_train_0 shape : ", y_train_0.shape,
                "\n","y_train_1 shape : ", y_train_1.shape,
                "\n","y_train_2 shape : ", y_train_2.shape)

        print("-----------------5----------------------")
        print(  "\n","y_test_0 shape : ", y_test_0.shape,
                "\n","y_test_1 shape : ", y_test_1.shape,
                "\n","y_test_2 shape : ", y_test_2.shape)
                
    return y_train_0, y_train_1, y_train_2


# import argparse

# def make_parser() -> argparse.ArgumentParser:
#     """Make parser"""
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--input_file_xyz",
#         type=str,
#         required=True,
#         help="Input file name, e.g. 'data_accelerometer.csv' or path_data/data.csv",
#     )
#     parser.add_argument(
#         "--input_file_labels",
#         type=str,
#         required=True,
#         help="Input file name, e.g. 'data_labels.csv' or path_data/data.csv"
#     )

#     return parser


# parser.add_argument("--input_file", type=str, help="Input file name")
# parser.add_argument("--output_file", type=str, help="Output file name")