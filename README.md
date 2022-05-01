## Sign Language
Identify hand gestures using machine learning and control a remote control car with the signs

Table of contents
=================

<!--ts-->
   * [Setup](#Setup)
   * [Installation](#Installation)
   * [Data](#Data)
   * [Project Tabs](#tabs)
      * [Predict](#Predict)
      * [Train](#Train)
      * [Capture](#Capture)
<!--te-->

## Setup

Create a directory for the virtualenv

```
mkdir RoboticsWork
```

Setup the virtual enviroment and install packages

```
python -m venv --system-site-packages .\RoboticsEnv\
```

Activate the enviroment

```
.\RoboticsEnv\Scripts\activate
```

ensure pip is the latest

```
pip install --upgrade pip
```

## Installation

Grab the work from git

```
git clone https://github.com/Amzo/Sign-Language.git
```

```
cd .\Sign-Language\
```

Ensure these are the latest to avoid opencv-python failing

```
pip install --upgrade setuptools wheel
```

Finally install the necessary packages

```
pip install -r requirements.txt
```

## Data

```
git clone https://github.com/Amzo/Sign-Language-Data.git
```

[NOTE] When training a model in the train tab, browse to the downloaded dataset. The setup of cuda if required is down to the user.
 
 
## Project Tabs
### Predict

This tab will allow connecting to a remote server and sending the predicted commands as 3bytes containing the pridicted character and the newline characters '\n'. e.G if A is sent to the server the server will receive 'A\n'. If there is no server to connect to this tab will not work, as it requires an active connection before making a prediction.

### Train

This tab allows fitting the convolutional neural network. Browse to the necessary data and select a model output folder. The training is done in 3 splits for a specified number of epochs. The model consists of an Xception model as a base for transfer learning, as well as a model for the fingerpoints from the csv file. The output layers of these two models are concatenated.

This tab also allows fitting the KNN with the CSV data file to generate a fitted model which is saved as a pickle file.

### Capture

The capture tab focuses on collecting data such as the images sabed as (100x100) pixels, as well as writing the finger datapoints to a csv files. The necessary directories for the data to be saved to are created automatically by the software.
