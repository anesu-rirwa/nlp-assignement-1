import streamlit as st

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import pickle

#allows us to load an image from a file as a PIL object
from keras.preprocessing.image import load_img 

#allows us to convert the PIL object into a NumPy array
from keras.preprocessing.image import img_to_array 

#prepare your image into the format the model requires. You should load images with the Keras load_img function so that you guarantee the images you load are compatible with the preprocess_input function.
from keras.applications.vgg16 import preprocess_input 

# models 
#pre-trained model we’re going to use
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans

#for reducing the dimensions of our feature vector
from sklearn.decomposition import PCA

from random import randint


header = st.container()
colab_file = st.container()
colab_link = '[Assignment Colab File Link](https://colab.research.google.com/drive/14cROjof2KWO9GfbzpAUpOqhwB7wBbu7M?usp=sharing)'


path = r"static\flower_images\flower_images"

# change the working directory to the path where the images are located
os.chdir(path)


with header:
    st.title('Image Cluster Webpage')
    st.text('Image clustering according to image similarity or the similarity of the texts that describe the images.')

with colab_file:
    st.header('Google Colab File Link')
    st.markdown(colab_link, unsafe_allow_html=True)
    #st.write(groups[1])

# this list holds all the image filename
flowers = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
  # loops through each file in the directory
    for file in files:
        if file.name.endswith('.png'):
          # adds only the image files to the flowers list
            flowers.append(file.name)

    # view the first 15 flower entries
    st.header('First 15 flower entries in the image directory')
    st.image(flowers[:15])

# load the image as a 224x224 array
img = load_img(flowers[0], target_size=(224,224))

# convert from 'PIL.Image.Image' to numpy array
img = np.array(img)

#st.write(img.shape)

reshaped_img = img.reshape(1,224,224,3)

#st.write(reshaped_img.shape)

x = preprocess_input(reshaped_img)

# load model
model = VGG16()
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# load the model first and pass as an argument
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

data = {}
p = r"static\flower_features.flower_features.pkl" # path to save the extracted features

# loop through each image in the dataset
for flower in flowers:
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(flower,model)
        data[flower] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(p,'wb') as file:
            pickle.dump(data,file)
          
 
# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))
st.header('Feature Shape')
feat.shape


# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1,4096)
st.header('Reshaped Feature Shape')
feat.shape


# get the unique labels (from the flower_labels.csv)
df = pd.read_csv('flower_labels.csv')
label = df['label'].tolist()
unique_labels = list(set(label))

pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

kmeans = KMeans(n_clusters=len(unique_labels), random_state=22)
kmeans.fit(x)

st.header('kmeans.labels_')
st.write(kmeans.labels_)

# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

# view the filenames in cluster 0
st.header('Cluster [0]')
st.image(groups[0])

st.header('Cluster [1]')
st.image(groups[1])