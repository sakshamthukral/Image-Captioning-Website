<h1>Step-By-Step Guide</h1>

<h3>Step 1: Import the following Libraries :-</h3>
import numpy as np <br>
import matplotlib.pyplot as plt <br>
import cv2 <br>
import keras <br>
import re <br>
import nltk <br>
from nltk.corpus import stopwords <br>
import string <br>
import json <br>
from time import time <br>
import pickle <br>
from keras.applications.vgg16 import VGG16 <br>
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions <br>
from keras.preprocessing import image <br>
from keras.models import Model,load_model <br>
from keras.preprocessing.sequence import pad_sequences <br>
from keras.utils import to_categorical  # it transforms the y-data into the vector format of categorical dataset <br>
from keras.layers import Input,Dense,Dropout,Embedding,LSTM <br>
from keras.layers.merge import add <br>

<h3>Step 2: Read text captions :- </h3> 
def readTextFile(path): <br>
    with open(path) as f: <br>
        captions = f.read() <br>
    return captions <br>
