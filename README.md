
# Quail Time Series Analysis / Quail Behavior Classifier / Quail Research Tools


![ethereal Bohemian Waxwing bird, Bombycilla garrulus :: intricate details, ornate, detailed illustration, octane render :: Johanna Rupprecht style, William Morris style :: trending on artstation --ar 9:16 - Midjourney v3 - Author ](https://storage.googleapis.com/dream-machines-output/a21d76a4-dc99-486f-9994-def7021bd6dd/0_0.png)


Quail are small birds that are commonly used in research studies due to their unique behavior and cognitive abilities. However, tracking and analyzing the behavior of quail can be challenging, especially when using traditional methods such as manual observation or video tracking. In this project, we present an open source solution for the detection and classification of quail behavior using time series data from accelerometers. By leveraging the power of machine learning and time series analysis, our approach enables the automatic and accurate tracking of quail behavior in a variety of research settings. With this tool, researchers can easily experiment with different machine learning models and feature engineering techniques to gain valuable insights into the behavior and cognitive abilities of quail. In addition, this tool can be used to test the effectiveness of different interventions or treatments on quail behavior.

## Repository Structure

This repository contains the following folders:

<pre>
├─── assets
│   ├─── images_1.png
│   ├─── *.png
│   └─── images_n.png
├─── data
│   ├─── accelerometer_data_x_y_z.csv
│   └─── labels.csv
├─── mlruns
│   ├─── .trash
│   ├─── experiment_default
│   └─── experiment_id
├───src
│   ├─── __init__.py
│   ├─── visualization_module.py
│   ├─── preprocessing_module.py
│   ├─── processing_module.py
│   ├─── utils_module.py
│   ├─── deep_learning_module.py
│   └─── streamlit_app.py
├─── main.ipynb
├─── main.py
├─── .gitignore
├─── poetry.lock
├─── pyproject.toml
├─── README.md
└─── requirements.txt
</pre>


## Table of Contents

1. Introduction: A brief overview of the project, including its purpose and goals.
2. Data: A description of the data used in this project, including its source and format.
3. Modules: A description of the modules used in this project, including their purpose and functionality.
4. Requirements: A list of the software and hardware requirements for using the tool.
5. Installation: Detailed instructions on how to install and set up the tool.
6. Usage: A description of how to use the tool, including any command-line arguments or configuration options.
7. Examples: A collection of sample inputs and outputs, along with instructions on how to run the examples.
8. API Reference: Detailed documentation on the various classes, methods, and functions provided by the tool.
9. Contributing: Guidelines for contributing to the project, including code conventions and how to submit pull requests.
10. License: Information on the open source license under which the project is released.


## 1. Introduction

This project is an open source tool for the detection and classification of quail behavior using accelerometer data. The goal of the tool is to enable researchers to track and analyze the behavior of quail in a variety of research settings. The tool includes several modules for preprocessing, processing, visualization, and deep learning, and it can be easily configured and customized to fit the specific needs of a research study. In addition, the tool integrates with MLFlow to provide experiment tracking and reproducibility, and it includes a graphical interface built with Streamlit for running experiments and exploring the results. Overall, this tool is a valuable resource for researchers interested in studying the behavior and cognitive abilities of quail.

## 2. Data

![asd](/V2/assets/images_1.png)

The data for this project was collected using accelerometers attached to the backs of quail. The accelerometers were placed in small backpacks and worn by the quail while they performed various behaviors. The accelerometer data was then recorded and labeled by researchers who watched videos of the quail and manually annotated each frame with the corresponding behavior.

The data consists of time series data from the accelerometers, as well as the corresponding labels for the quail behavior. The behavior labels include three classes: normal behavior, reproductive behavior, and grooming behavior. The data was collected from multiple quail in different environments and under different conditions, providing a diverse and representative sample of quail behavior. The data is available in the data folder of this repository.


## 3. Modules

### 3.1. Visualization Module
![asd](/V2/assets/desequilibre.png)
![asd](/V2/assets/signal_segment.png)
### 3.2. Preprocessing Module

![asd](/V2/assets/split_windows.png)

![asd](/V2/assets/val_abs.png)

### 3.3. Processing Module
![asd](/V2/assets/envolve.png)
### 3.4. Utils Module

### 3.5. Deep Learning Module
![asd](/V2/assets/like_image.png)


## 4. Results

![asd](/V2/assets/first_results.png)
![asd](/V2/assets/confusion_matrix.png)
"# quail_behavior_classifier" 
