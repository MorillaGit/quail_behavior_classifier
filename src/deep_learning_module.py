import numpy as np
import mlflow.tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import AUC


def reshape_data_cnn(
    train: np.ndarray = None,
    test: np.ndarray = None,
    arr: np.ndarray = None,
    debug: bool = False,
) -> tuple:
    """This function takes in a tensor of data in a specific shape and modifies it
    to be compatible with a convolutional neural network. This may include adding
    an additional dimension to the tensor, changing the order of the dimensions,
    or flattening the tensor. Convert the data in format  (samples, rows, cols) to the format (samples, rows, cols, dimension_added)

    :param train: np.array to convert, defaults to None
    :type train: np.ndarray, optional
    :param test: np.array to convert, defaults to None
    :type test: np.ndarray, optional
    :param arr: np.array to convert, defaults to None
    :type arr: np.ndarray, optional
    :param debug: is a flag to know if the function is in debug mode, defaults to False
    :type debug: bool, optional
    :return: return a array with the data converted in format (samples, rows, cols, dimension_added)
    :rtype: tuple
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
            print("\n", "train shape : ", train.shape)

        if test is not None:
            print("------------test reshape---------------")
            print("\n", "test shape : ", test.shape)

        if arr is not None:
            print("------------arr reshape---------------")
            print("\n", "arr shape : ", arr.shape)

    return train, test


def run_experiment_cnn(
    input_shape: tuple,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str = "Default CNN",
    epochs: int = 10,
    # batch_size=32, # better results without batch size
    debug: bool = False,
    learning_rate: float = 0.001,
    metrics: list = ["accuracy"],
    num_classes: int = 3,
    loss: str = "sparse_categorical_crossentropy",
    filters_cnn_base: list = [8, 16, 16],
    kernel_size_cnn_base: list = [(3, 3), (3, 3), (3, 3)],
    pool_size_cnn_base: list = [(1, 1), (1, 1), (1, 1)],
    dropout_rate: list = [0.2, 0.4, 0.6],
    num_units_dense_base: list = [128, 64],
    padding_exp: str = "same",
):

    #TODO all list to be None and add to code

    """This function runs an experiment with a convolutional neural network model, it uses the mlflow library to log the results of the experiment, and is possible modify the parameters of the model

    :param input_shape: shape of the input data
    :type input_shape: tuple
    :param X_train: training data
    :type X_train: np.ndarray
    :param y_train: training labels
    :type y_train: np.ndarray
    :param X_test: test data
    :type X_test: np.ndarray
    :param y_test: test labels
    :type y_test: np.ndarray
    :param experiment_name: name of the experiment, defaults to "Default CNN"
    :type experiment_name: str, optional
    :param epochs: number of epochs, defaults to 10
    :type epochs: int, optional
    :param debug: is a flag to know if the function is in debug mode, defaults to False
    :type debug: bool, optional
    :param learning_rate: learning rate, defaults to 0.001
    :type learning_rate: float, optional
    :param metrics: metrics to evaluate the model, defaults to ["accuracy"]
    :type metrics: list, optional
    :param num_classes: number of classes, defaults to 3
    :type num_classes: int, optional
    :param loss: loss function, defaults to "sparse_categorical_crossentropy"
    :type loss: str, optional
    :param filters_cnn_base: number of filters for each convolutional layer, defaults to [8, 16, 16]
    :type filters_cnn_base: list, optional
    :param kernel_size_cnn_base: kernel size for each convolutional layer, defaults to [(3, 3), (3, 3), (3, 3)]
    :type kernel_size_cnn_base: list, optional
    :param pool_size_cnn_base: pool size for each convolutional layer, defaults to [(1, 1), (1, 1), (1, 1)]
    :type pool_size_cnn_base: list, optional
    :param dropout_rate: dropout rate for each dense layer, defaults to [0.2, 0.4, 0.6]
    :type dropout_rate: list, optional
    :param num_units_dense_base: number of units for each dense layer, defaults to [128, 64]
    :type num_units_dense_base: list, optional
    :param padding_exp: padding for each convolutional layer, defaults to "same"
    :type padding_exp: str, optional
    """

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    with mlflow.start_run() as run:

        # set name experiment
        mlflow.set_experiment(experiment_name)

        def create_model_cnn_basic(
            input_shape_dataset: tuple = input_shape, num_classes: int = num_classes
        ) -> tf.keras.Model:

            """This function creates a basic convolutional neural network model with 2 convolutional layers, 2 dense layers and a softmax layer

            :param input_shape_dataset: shape of the input data
            :type input_shape_dataset: tuple
            :param num_classes: number of classes
            :type num_classes: int
            :return: return a model
            :rtype: tf.keras.Model
            """

            if debug:
                print("------------model summary---------------")
                print("input_shape_dataset", input_shape_dataset)
                print("num_classes", num_classes)

            input_shape_dataset: tuple
            model = Sequential()
            model.add(
                Conv2D(
                    filters_cnn_base[0],
                    kernel_size=kernel_size_cnn_base[0],
                    activation="relu",
                    input_shape=input_shape_dataset,
                    padding=padding_exp,
                    # strides=2,
                )
            )
            model.add(MaxPooling2D(pool_size=pool_size_cnn_base[0]))
            model.add(
                Conv2D(
                    filters_cnn_base[1],
                    kernel_size=kernel_size_cnn_base[1],
                    activation="relu",
                )
            )
            model.add(MaxPooling2D(pool_size=pool_size_cnn_base[1]))
            model.add(
                Conv2D(
                    filters_cnn_base[2],
                    kernel_size=kernel_size_cnn_base[2],
                    activation="relu",
                )
            )
            model.add(MaxPooling2D(pool_size=pool_size_cnn_base[2]))
            model.add(Dropout(dropout_rate[0]))
            model.add(Flatten())
            model.add(Dense(num_units_dense_base[0], activation="relu"))
            model.add(Dropout(dropout_rate[1]))
            model.add(Dense(num_units_dense_base[1], activation="relu"))
            model.add(Dropout(dropout_rate[2]))
            model.add(Dense(num_classes, activation="softmax"))
            if debug:
                model.summary()
            return model

        model = create_model_cnn_basic()

        mlflow.log_param("input_shape_dataset", input_shape)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("kernel_size_cnn_base", kernel_size_cnn_base)
        mlflow.log_param("pool_size_cnn_base", pool_size_cnn_base)
        mlflow.log_param("filter cnn base", filters_cnn_base)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("num_units_dense_base", num_units_dense_base)
        mlflow.log_param("padding", padding_exp)

        # save summary of the model
        # with open("model_summary.txt", "w") as fh:
        #     model.summary(print_fn=lambda line: fh.write(line + "\n"))

        # create the compile
        model.compile(
            optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=metrics
        )

        with mlflow.start_run():
            # mlflow.log_param("optimizer", optimizer)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("metrics", metrics)
            mlflow.log_param("loss", loss)

        # log interpolation
        # mlflow.log_param("interpolation", interpolation)

        # fit model
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            #  batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0,
        )

        mlflow.tensorflow.log_model(model, "model")
        mlflow.log_param("epochs", epochs)

        mlflow.log_metric("loss", history.history["loss"][-1])
        mlflow.log_metric("accuracy", history.history["accuracy"][-1])
        mlflow.log_metric("val_loss", history.history["val_loss"][-1])
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])

        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test", "Loss", "Val_loss"], loc="upper left")
        plt.grid()

        plt.savefig(
            "Accuracy.png", dpi=300
        )  # no se exactamento donde lo guarda por lo que no puedo guardar en articafts
        mlflow.log_artifact("Accuracy.png")  # Esta linea no entrega falso

        # make a prediction
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        # save y_pred to csv
        # np.savetxt('y_pred.csv', y_pred, delimiter=',')
        # mlflow.log_artifact('y_pred.csv')

        cm = confusion_matrix(y_test, y_pred)
        _, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", linewidths=0.5, square=True, cmap="Blues_r", ax=ax
        )
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        plt.savefig("confusion_matrix.png", dpi=300)
        mlflow.log_artifact("confusion_matrix.png")

        if debug:
            print("run_id: {}".format(run.info.run_id))
            print("artifacts_uri: {}".format(run.info.artifact_uri))
    return history


    
def run_experiment_rnn(
    input_shape: tuple,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str = "Default RNN",
    epochs: int = 10,
    # batch_size=32, # better results without batch size
    debug: bool = False,
    learning_rate: float = 0.001,
    metrics: list = ["AUC()"],
    num_classes: int = 3,
    loss: str = "sparse_categorical_crossentropy",
    num_units_lstm_base: int = 32,
    dropout_rate: float = 0.2,
    num_units_dense_base: int = 32,
    padding_exp: str = "same",

):

    #TODO all list to be None and add to code

    """This function runs an experiment with a recurrent neural network model, it uses the mlflow library to log the results of the experiment, and is possible modify the parameters of the model

   
    """

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    with mlflow.start_run() as run:

        # set name experiment
        mlflow.set_experiment(experiment_name)

        def create_model_lstm_basic(
            input_shape_dataset: tuple = input_shape, num_classes: int = num_classes
        ) -> tf.keras.Model:

            """This function creates a basic convolutional neural network model with 2 convolutional layers, 2 dense layers and a softmax layer

            :param input_shape_dataset: shape of the input data
            :type input_shape_dataset: tuple
            :param num_classes: number of classes
            :type num_classes: int
            :return: return a model
            :rtype: tf.keras.Model
            """

            if debug:
                print("------------model summary---------------")
                print("input_shape_dataset", input_shape_dataset)
                print("num_classes", num_classes)

            input_shape_dataset: tuple
            model = Sequential()
            model.add(
                LSTM(
                    num_units_lstm_base,
                    input_shape=input_shape_dataset,
                    return_sequences=True,
                )
            )
            return model

        model = create_model_lstm_basic()

        # mlflow.log_param("input_shape_dataset", input_shape)
        mlflow.log_param("num_classes", num_classes)
        # mlflow.log_param("kernel_size_cnn_base", kernel_size_cnn_base)
        # mlflow.log_param("pool_size_cnn_base", pool_size_cnn_base)
        # mlflow.log_param("filter cnn base", filters_cnn_base)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("num_units_dense_base", num_units_dense_base)
        mlflow.log_param("padding", padding_exp)

        # save summary of the model
        # with open("model_summary.txt", "w") as fh:
        #     model.summary(print_fn=lambda line: fh.write(line + "\n"))

        # create the compile
        model.compile(
            optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=metrics
        )

        # mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("metrics", metrics)
        mlflow.log_param("loss", loss)

        # log interpolation
        # mlflow.log_param("interpolation", interpolation)

        # fit model
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            #  batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0,
        )

        mlflow.tensorflow.log_model(model, "model")
        mlflow.log_param("epochs", epochs)

        mlflow.log_metric("loss", history.history["loss"][-1])
        mlflow.log_metric("accuracy", history.history["accuracy"][-1])
        mlflow.log_metric("val_loss", history.history["val_loss"][-1])
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])

        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test", "Loss", "Val_loss"], loc="upper left")
        plt.grid()

        plt.savefig(
            "Accuracy.png", dpi=300
        )  # no se exactamento donde lo guarda por lo que no puedo guardar en articafts
        mlflow.log_artifact("Accuracy.png")  # Esta linea no entrega falso

        # make a prediction
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        # save y_pred to csv
        # np.savetxt('y_pred.csv', y_pred, delimiter=',')
        # mlflow.log_artifact('y_pred.csv')

        cm = confusion_matrix(y_test, y_pred)
        _, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", linewidths=0.5, square=True, cmap="Blues_r", ax=ax
        )
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        plt.savefig("confusion_matrix.png", dpi=300)
        mlflow.log_artifact("confusion_matrix.png")

        if debug:
            print("run_id: {}".format(run.info.run_id))
            print("artifacts_uri: {}".format(run.info.artifact_uri))
    return history