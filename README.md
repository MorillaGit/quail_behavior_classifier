
<details>
<summary>traslations:</summary>
- [Español](traslations/README-sp.md)
</details>



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
├─── traslations
│   ├─── READNE-sp.md
├───src
│   ├─── __init__.py
│   ├─── preprocessing.py
│   ├─── processing.py
│   ├─── utils.py
│   └─── deep_learning_module.py
├─── main.ipynb
├─── make_predictions.ipynb
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
4. Installation: Detailed instructions on how to install and set up the tool.
5. Usage: A description of how to use the tool, including any command-line arguments or configuration options.
6. Examples: A collection of sample inputs and outputs, along with instructions on how to run the examples.
7. License: Information on the open source license under which the project is released.


## 1. Introduction

This project is an open source tool for the detection and classification of quail behavior using accelerometer data. The goal of the tool is to enable researchers to track and analyze the behavior of quail in a variety of research settings. The tool includes several modules for preprocessing, processing, visualization, and deep learning, and it can be easily configured and customized to fit the specific needs of a research study. In addition, the tool integrates with MLFlow to provide experiment tracking and reproducibility, and it includes a graphical interface built with Streamlit (developing) for running experiments and exploring the results. Overall, this tool is a valuable resource for researchers interested in studying the behavior and cognitive abilities of quail.

## 2. Data

![asd](assets/images_1.png)

The data for this project was collected using accelerometers attached to the backs of quail. The accelerometers were placed in small backpacks and worn by the quail while they performed various behaviors. The accelerometer data was then recorded and labeled by researchers who watched videos of the quail and manually annotated each frame with the corresponding behavior.

The data consists of time series data from the accelerometers, as well as the corresponding labels for the quail behavior. The behavior labels include three classes: normal behavior, reproductive behavior, and grooming behavior. The data was collected from multiple quail in different environments and under different conditions, providing a diverse and representative sample of quail behavior. The data is available in the data folder of this repository.


... in progress ...