import streamlit as st
import tensorflow as tf


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('./models/fingerprint_model.h5')
  return model
model=load_model()
file = st.file_uploader("Upload an image of a fingerprint", type=["jpg", "png", "bmp"])
import cv2
from PIL import Image, ImageOps
import numpy as np

classes = ['ben_afflek','elton_john','jerry_seinfeld', 'madonna','mindy_kaling']

def import_and_predict(image_data, model):    
        img_size=(224,224)
        image = ImageOps.fit(image_data, img_size, Image.ANTIALIAS)
        img = np.expand_dims(image_data, axis=0)
        prediction = model.predict(img)
        return prediction
        
if file is None:
    st.text("Please upload an image file.")
else:
    img = Image.open(file)
    st.image(img, use_column_width=True)  
    """img = cv2.resize(img,(224,224))
                images_arr = np.asarray(img)
                images_arr = images_arr.astype('float32')
                images_arr = images_arr.reshape(-1, 224,224, 1)"""
    predictions = import_and_predict(img, model)
    score = tf.nn.softmax(predictions[0])
    index = np.argmax(predictions[0])
    finger_predicted_class = classes[index]
    probability = predictions[0][index]*100
st.write("This fingerprint belongs to ",finger_predicted_class," with a ",probability," percent confidence.")




"""
#FACE
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # opencv
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from keras.models import load_model
from PIL import Image

import os
# load the facenet model
facenet_model = load_model('./models/facenet_keras.h5')
print('Loaded Model')
from joblib import dump, load
loaded_model = load('./models/facemodel.joblib')
from sklearn.preprocessing import LabelEncoder
## Load the trained LabelEncoder and SVM model
out_encoder=LabelEncoder()
loaded_encoder = out_encoder
loaded_predictor = loaded_model
loaded_facenet = facenet_model
labels=['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna','mindy_kaling']
loaded_encoder.fit(labels)
LabelEncoder()
loaded_encoder.classes_
array(['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling'], dtype='<U14')

# Function to get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

# Single Function capable to process input image and return the bounding box dimensions an
def face_recognize(image):
    ## Extract the face and bounding box dimensions from the image by using pretrained MTC
    faces, box_dimensions = extract_face(image_pixels=image)
    X = np.asarray(faces)
    ## Get the Face Embeddings for the extracted face pixels and store as numpy array
    embedding = get_embedding(loaded_facenet, X)
    X = []
    X.append(embedding)
    X = np.asarray(X)
    ## Predict label for the face by using the pretrained models
    prediction = loaded_predictor.predict(X)
    predicted_label = loaded_encoder.inverse_transform([prediction])
    return predicted_label[0], box_dimensions

def extract_face(filename=None, image_pixels=None, required_size=(160, 160)):
    if filename is not None:
        image = Image.open(filename)
        image = image.convert('RGB')
        pixels = np.asarray(image)
    elif image_pixels is not None:
        pixels = image_pixels
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    box_dimensions = (x1, y1, width, height)
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array, box_dimensions
## Testing the model with the random image
filename = 'datasets/face/val/madonna/httpcdncdnjustjaredcomwpcontent'
image = Image.open(filename)
image = image.convert('RGB')
pixels = np.asarray(image)
face_predicted_class = face_recognize(pixels)[0]
print("This face belongs to ",face_predicted_class, ".")
"""