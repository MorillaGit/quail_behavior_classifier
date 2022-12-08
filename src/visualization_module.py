
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# this function receives a dataframe , and two integers optionally and plots the data
def plot_data(  dataframe: pd.DataFrame,
                start: int = 0, 
                end: int = None, 
                graph_size : tuple = (20,5),
                title: str = "Plot time series"
                # ,background_plt: str = 'dark_background'
                ) -> None:


    """ Plot the data from the dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
        data to plot
    start : int, optional
        start index, by default 0
    end : int, optional
        end index, by default df.shape[0]
    title : str, optional
        title of the plot, by default "Algo"
    background : str, optional
        background of the plot, by default 'dark_background', more information: https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        list of background: "Default"
                            "Classic"
                            "Solarize_Light2"
                            "bmh"
                            "dark_background"
                            "fast"
                            "fivethirtyeight"
                            "ggplot"
                            "grayscale"
                            "seaborn"
                            "seaborn-bright"
                            "seaborn-colorblind" etc...

    Returns
    -------
    Graph
        plot of the data

    """

    # Set dark background in matplotlib
    # plt.style.use(background_plt) TODO: fix this

    # print("Plotting data...")

    if end == None:
        end = dataframe.shape[0]

    # plot the dataframe
    dataframe[start:end].plot(subplots=True, figsize=graph_size, title=title)
    plt.show()


# TODO: dont tested
# this function receives a array , and two integers optionally and plots the data
def plot_arr(arr: np.array , start: int = 0, end: int = None, title: str=" Plot ") -> None: 

    """ Plot the data from the dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
         data to plot
    start : int, optional
        start index, by default 0
    end : int, optional
        end index, by default df.shape[0]
    """
    # plot the data
    if end is None:
        end = df.shape[0]
    
    plt.show()
    plt.figure(figsize=(12,5.8))
    plt.title(title,fontsize=16)
    # change the color of the line
    plt.axhline(y=0, color='w')
    plt.plot(np.arange(0,len(arr[start:end]),1),arr[start:end],linewidth=0.5, color=(0.8,0,0.8))
    plt.show()

# TODO: dont tested
# this funcion receives a two integers and plot the data
# def plot_comparative(start: int = 0, end: int = df.shape[0], title: str = "plot comparative") -> None: 
#         """ Plot the data from the dataframe, to see relationship between the data and labels
        
#         Parameters
#         ----------
#         start : int, optional
#             start index, by default 0
#         end : int, optional
#             end index, by default df.shape[0]
#         """
    
#         # plot the data
            

#         plt.figure(figsize=(12,5.8))
#         plt.title(title,fontsize=16)
#         plt.axhline(y=0, color='w')
#         plt.plot(np.arange(0,len(data_arr[start:end]),1),data_arr[start:end],linewidth=0.5, color=(0.2,0.5,0.8,1))
#         plt.plot(np.arange(0,len(label_arr[start:end]),1),label_arr[start:end],linewidth=5, color=(0.9,0.1,0.8,1))
#         plt.show()
#         return None


# plot envelope TODO dosent work
def plot_interpolation_labeled( data: np.ndarray,
                                label: np.ndarray, 
                                sampling_rate: int, 
                                start: int = 0, 
                                end: int = df.shape[0], 
                                title: str = "plot interpolation labeled") -> None:
    
        """ Plot the data from the dataframe, to see relationship between the data and labels
        
        Parameters
        ----------
        data : np.ndarray
            data to plot
        label : np.ndarray
            label to plot
        sampling_rate : int
            sampling rate
        start : int, optional
            start index, by default 0
        end : int, optional
            end index, by default df.shape[0]
        """
        x  = np.arange(len(data))
        print(x.shape)

        for channel in range(data.shape[1]):
            # calculate the envelope
            sig = data.T[channel]
            inter = envelope(data, sampling_rate)

            # plot the data
        plt.figure(figsize=(12,5.8))
        plt.title(title,fontsize=16)
        plt.axhline(y=0, color='w')
        
        plt.plot(x[start:end],inter(x[start:end]),linewidth=3, color=(0.2,0.5,0.8,1))
        plt.plot(x[start:end],sig[start:end],linewidth=5, color=(0.9,0.1,0.8,1))
        plt.plot(x[start:end],label[start:end],linewidth=0.4, color=(0.8,0.1,0.8,1))
        plt.show()
        return None


# 11_ module_visualization TODO
# this function plot the data whit vertical lines defining by parameter