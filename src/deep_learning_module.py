
import numpy as np
import pandas as pd
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import mlflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.optimizers import Adam
from src import deep_learning_module
import importlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score
import mlflow
import datetime


def reshape_data_cnn(
                train : np.ndarray = None, 
                test : np.ndarray = None,
                arr : np.ndarray = None, 
                debug: bool = False
                ) -> tuple:
    """_summary_

    Args:
        train (np.ndarray): _description_
        test (np.ndarray): _description_
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        tuple: _description_
    """
    # reshape data
    if train is not None:
        train = train.reshape(train.shape[0], train.shape[1], train.shape[2], 1)
    if test is not None:
        test = test.reshape(test.shape[0], test.shape[1], test.shape[2], 1)
    if arr is not None:
        arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)

    if debug:
        if train is not None:
            print("------------train reshape-----------------")
            print(  "\n","train shape : ", train.shape)

        if test is not None:
            print("------------test reshape---------------")
            print(  "\n","test shape : ", test.shape)

        if arr is not None:
            print("------------arr reshape---------------")
            print(  "\n","arr shape : ", arr.shape)


    return train, test

def create_model_cnn_basic(input_shape_dataset, num_classes, debug=False):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape_dataset, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 1),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    if debug:
        model.summary()
    return model

def create_model_parametric(input_shape_dataset : tuple,
                            num_classes : int,
                            debug : bool = False,
                            filters_base : int = 32,                            
                            ) -> tf.keras.Model:
    """_summary_

    Args:
        input_shape_dataset (tuple): _description_
        num_classes (int): _description_
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        tf.keras.Model: _description_
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape_dataset, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 1),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    if debug:
        model.summary()
    return model
# TODO not working
def build_cnn_complex(
            input_shape : tuple,
            num_classes : int,
            debug: bool = False,

            num_layers_conv : int = 2,
            num_layers_dense : int = 2,

            num_filters_1 : int = 32,
            num_filters_2 : int = 64,
            num_filters_3 : int = 128,

            kernel_size_1 : int = (3,3),
            kernel_size_2 : int = (4,4),
            kernel_size_3 : int = (5,5),

            pool_size_0 : int = (2,2),
            pool_size_1 : int = (2,2),
            pool_size_2 : int = (4,4),


            num_units_1 : int = 128,
            num_units_2 : int = 64,
            num_units_3 : int = 32,

            dropout_rate : float = 0.25,
            activation : str = "relu",

            ) -> Sequential:

    """Creates a convolutional neural network model with the specified architecture.
    
    Args:
        num_layers: Number of layers in the neural network.
        num_units: Number of units per layer in the neural network.
        activation: Activation function used by the layers in the neural network.
        kernel_size: Kernel size used by the layers in the neural network.
        dropout_rate: Percentage of units to drop out in the layers of the neural network.
        ...: Other arguments for defining the architecture of the neural network.
        
    Returns:
        A convolutional neural network model with the specified architecture.
    """
    tf.compat.v1.logging.set_log_device_placement(False)


    model = Sequential()
    if num_layers_conv == 1:
        model.add(tf.keras.layers.Conv2D(num_filters_1, kernel_size_1, activation=activation, input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size_0))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    elif num_layers_conv == 2:
        model.add(tf.keras.layers.Conv2D(num_filters_1, kernel_size_1, activation=activation, input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size_0))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Conv2D(num_filters_2, kernel_size_2, activation=activation,padding="same"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size_1))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    elif num_layers_conv == 3:
        model.add(tf.keras.layers.Conv2D(num_filters_1, kernel_size_1, activation=activation, input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size_0))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Conv2D(num_filters_2, kernel_size_1, activation=activation,padding="same"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size_1))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        # TODO solve this problem
        # model.add(tf.keras.layers.Conv2D(num_filters_3, kernel_size_1, activation=activation,padding="same"))
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size_2))
        # model.add(tf.keras.layers.Dropout(dropout_rate))

    if num_layers_dense == 1:
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(num_units_1, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    elif num_layers_dense == 2:
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(num_units_1, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(num_units_2, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    elif num_layers_dense == 3:
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(num_units_1, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(num_units_2, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(num_units_3, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))


    if debug:
        print("------------model summary---------------")
        model.summary()
    
    return model




def run_experiment_cnn(
    input_shape,
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    epochs=10, 
    batch_size=32, 
    debug=False, 
    optimizer = Adam, 
    learning_rate = 0.005, 
    metrics = ['accuracy'], 
    num_classes = 3,
    loss = 'sparse_categorical_crossentropy'):


    with mlflow.start_run() as run:

        model = deep_learning_module.create_model_cnn_basic(input_shape, num_classes, debug=False)

        mlflow.log_param("input_shape_dataset", input_shape)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("num_layers_conv", 1)
        mlflow.log_param("num_layers_dense", 1)

        # create the compile
        model.compile(  optimizer=Adam(learning_rate=learning_rate),
                         loss=loss, 
                         metrics=metrics)


        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("metrics", metrics)
        mlflow.log_param("loss", loss)

        # log interpolation
        # mlflow.log_param("interpolation", interpolation)

        # fit model
        history = model.fit(X_train, y_train,
                         epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(X_test, y_test),
                         verbose=0)
        # log params
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        # fit model
        history = model.fit(    X_train, 
                                y_train,    
                                epochs=epochs, 
                                batch_size=batch_size, 
                                validation_data=(X_test, y_test),
                                verbose=0)
        
        # log model
        mlflow.tensorflow.log_model(model, "model")

        # log metrics
        mlflow.log_metric("loss", history.history['loss'][-1])
        mlflow.log_metric("accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("val_loss", history.history['val_loss'][-1])
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])

        # log artifacts TODO

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        # grid on
        plt.grid()        
        	
        plt.savefig('Accuracy.png', dpi=300)  # no se exactamento donde lo guarda por lo que no puedo guardar en articafts
        mlflow.log_artifact('Accuracy.png') # Esta linea no entrega falso

        # make a prediction
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        # save y_pred to csv
        # np.savetxt('y_pred.csv', y_pred, delimiter=',')
        # mlflow.log_artifact('y_pred.csv')

        # log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(cm, annot=True, fmt="d");
        plt.figure(figsize=(10,10))
        plt.title("Confusion matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png', dpi=300)
        mlflow.log_artifact('confusion_matrix.png')

        # save the model trained, with name model "interpotation value" .h5, and timestamp
        # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # model.save("model"+str(interpolation)+timestamp+".h5")
        # model.save('model.h5')

        # curve roc TODO



        if debug:
            print("run_id: {}".format(run.info.run_id))
            print("artifacts_uri: {}".format(run.info.artifact_uri))
    return history