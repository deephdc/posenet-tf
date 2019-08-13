DEEP Open Catalogue: Speech to Text
====================================

[![Build Status](https://jenkins.indigo-datacloud.eu:8080/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/speech-to-text-tf/master)](https://jenkins.indigo-datacloud.eu:8080/job/Pipeline-as-code/job/DEEP-OC-org/job/speech-to-text-tf/job/master/)


**Author:** [Lara Lloret Iglesias](https://github.com/laramaktub) (CSIC)

**Project:** This work is part of the [DEEP Hybrid-DataCloud](https://deep-hybrid-datacloud.eu/) project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 777435.

This is a plug-and-play tool to train and evaluate a speech to text tool using deep neural networks. The network architecture is based in one of the tutorials provided by Tensorflow (https://www.tensorflow.org/tutorials/sequences/audio_recognition).
The architecture used in this tutorial is based on some described in the paper [Convolutional Neural Networks for Small-footprint Keyword Spotting](https://static.googleusercontent.com/media/research.google.com/es//pubs/archive/43969.pdf). It was chosen because it's comparatively simple, quick to train, and easy to understand, rather than being state of the art. There are lots of different approaches to building neural network models to work with audio, including recurrent networks or dilated (atrous) convolutions. This tutorial is based on the kind of convolutional network that will feel very familiar to anyone who's worked with image recognition. That may seem surprising at first though, since audio is inherently a one-dimensional continuous signal across time, not a 2D spatial problem. We define a window of time we believe our spoken words should fit into, and converting the audio signal in that window into an image. This is done by grouping the incoming audio samples into short segments, just a few milliseconds long, and calculating the strength of the frequencies across a set of bands. Each set of frequency strengths from a segment is treated as a vector of numbers, and those vectors are arranged in time order to form a two-dimensional array. This array of values can then be treated like a single-channel image, and is known as a spectrogram. An example of what one of these spectrograms looks like:

<p align="center">
<img src="./reports/figures/spectrogram.png" alt="spectrogram" width="400">
</p>

To start using this framework run:

```bash
git clone https://github.com/deephdc/speech-to-text-tf
cd speech-to-text-tf
pip install -e .
```

 **Requirements:**
 
- This project has been tested in Ubuntu 18.04 with Python 3.6.5. Further package requirements are described in the `requirements.txt` file.
- It is a requirement to have [Tensorflow>=1.12.0 installed](https://www.tensorflow.org/install/pip) (either in gpu or cpu mode). This is not listed in the `requirements.txt` as it [breaks GPU support](https://github.com/tensorflow/tensorflow/issues/7166). 
- Run `python -c 'import cv2'` to check that you installed correctly the `opencv-python` package (sometimes [dependencies are missed](https://stackoverflow.com/questions/47113029/importerror-libsm-so-6-cannot-open-shared-object-file-no-such-file-or-directo) in `pip` installations).

## Project Organization


    ├── LICENSE
    ├── README.md              <- The top-level README for developers using this project.
    ├── data
    │   ├── audios                <- The original, immutable data dump.
    │   │
    │   └── data_splits            <- Scripts to download or generate data
    │
    ├── docs                   <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── docker                 <- Directory for Dockerfile(s)
    │    ├── models                 <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks              <- Jupyter notebooks. 
    │
    ├── references             <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures            <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
    │                             generated with `pip freeze > requirements.txt`
    ├── test-requirements.txt  <- The requirements file for the test environment
    │
    ├── setup.py               <- makes project pip installable (pip install -e .) so imgclas can be imported
    ├── posenetclas    <- Source code for use in this project.
    │   ├── __init__.py        <- Makes imgclas a Python module
    │   │
    │   ├── dataset            <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features           <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models             <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── model.py
    │   │
    │   └── tests              <- Scripts to perfrom code testing + pylint script
    │   │
    │   └── visualization      <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini                <- tox file with settings for running tox; see tox.testrun.org

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Workflow

### 1. Data preprocessing

The first step to train your speech to text neural network is to put your .wav files into folders. The name of each folder should correspond to the label for those particular audios.  

#### 1.1 Prepare the audios

Put your audios in the`./data/dataset_files` folder. If you are using the DEEP api, you can also provide an URL with the location of the tar.gz containing all the folders with the training files. This will automatically download the tar.gz, read the labels and get everything ready to start the training. 

Please use wav files. 


### 2. Train the classifier

Before training the classifier you can customize the default parameters of the configuration file. To have an idea of what parameters you can change, you can explore them using the [dataset exploration notebook](./notebooks/1.0-Dataset_exploration.ipynb). This step is optional and training can be launched with the default configurarion parameters and still offer reasonably good results.

Once you have customized the configuration parameters in the  `./etc/config.yaml` file you can launch the training running `./imgclas/train_runfile.py`. You can monitor the training status using Tensorboard.

After training you can check training statistics and check the logs where you will be able to find the standard output during the training together with the confusion matrix after the training was finished.

Since usually this type of models are used in mobile phone application, the training generates the model in .pb format allowing to use it easily to perfom inference from a mobile phone app.

### 3. Test the classifier

You can test the classifier on a number of tasks: predict a single local wav file (or url) or predict multiple wavs (or urls). 


You can also make and store the predictions of the `test.txt` file (if you provided one). Once you have done that you can visualize the statistics of the predictions like popular metrics (accuracy, recall, precision, f1-score), the confusion matrix, etc by running the [predictions statistics notebook](./notebooks/3.1-Prediction_statistics.ipynb).


Finally you can launch a simple web page to use the trained classifier to predict audios (both local and urls) on your favorite brownser.


## Launching the full API

#### Preliminaries for prediction

If you want to use the API for prediction,  you have to do some preliminary steps to select the model you want to predict with:

- copy your desired `.models/[timestamp]` to `.models/api`. If there is no `.models/api` folder, the default is to use the last available timestamp.
- in the `.models/api/ckpts` leave only the desired checkpoint to use for prediction. If there are more than one chekpoints, the default is to use the last available checkpoint.

#### Running the API


To access this package's complete functionality (both for training and predicting) through an API you have to install the [DEEPaaS](https://github.com/indigo-dc/DEEPaaS) package:

```bash
git clone https://github.com/indigo-dc/deepaas
cd deepaas
pip install -e .
```

and run `deepaas-run --listen-ip 0.0.0.0`.
From there you will be able to run training and predictions of this package  using `model_name=posenetclas`.

<img src="./reports/figures/deepaas.png" alt="deepaas" width="1000"/>

