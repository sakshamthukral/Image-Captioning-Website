#!/usr/bin/env python
# coding: utf-8

# #### If a completely new image comes,our this module will be able to generate the caption even for that image

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras
import json
import tensorflow as tf
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input,Dense,Dropout,Embedding,LSTM
from keras.layers.merge import add
import warnings
warnings.filterwarnings("ignore")


# In[20]:

model = load_model("model_weights/model_9.h5")
model._make_predict_function() # to prevent Value error tensor tensor in tensor flow, this function tell the model to use the current backend ie tensorflow to load the model

print("Main Model Loaded")
# In[21]:


model_temp = ResNet50(weights="imagenet",input_shape=(224,224,3))

model_resnet = Model(model_temp.input,model_temp.layers[-2].output)
print(model_temp.layers[-2].output)
model_resnet._make_predict_function()  # to prevent Value error tensor tensor in tensor flow, this function tell the model to use the current backend ie tensorflow to load the model

print("Resnet Model Loaded Successfully")

# In[23]:

print("Sessions Created")
def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img


# In[24]:


def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1]) # we want the shape to be (1,2048)
    return feature_vector

with open("./storage/word_to_idx.pkl",'rb') as w2i:
    word_to_idx = pickle.load(w2i)
with open("./storage/idx_to_word.pkl",'rb') as i2w:
    idx_to_word = pickle.load(i2w)

def predict_caption(photo):
    in_text = "startseq"
    max_len = 38
    for i in range(max_len):

        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen = max_len,padding='post')
        print(sequence)
        ypred = model.predict([photo,sequence])
        word = idx_to_word[ypred.argmax()]
        in_text += ' ' + word
    
        if word == 'endseq':
            break
            
    # Final caption will be the in_text after removing 'startseq' and 'endseq'
    final_caption = in_text.split()[1:-1] # removing staring word that is startseq and the ending word that os endseq
    final_caption = ' '.join(final_caption)

    return final_caption

# In[32]:
def caption_this_image(image):
    enc = encode_image(image)

    in_text = "startseq"
    max_len = 38
    
    in_text = predict_caption(enc)
    # Final caption will be the in_text after removing 'startseq' and 'endseq'
    final_caption = in_text.split()[1:-1] # removing staring word that is startseq and the ending word that os endseq
    final_caption = ' '.join(final_caption)
    
    return final_caption

print(caption_this_image("../captioning_bot/static/images.jpg"))

