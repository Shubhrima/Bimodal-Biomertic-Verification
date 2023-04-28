from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import tensorflow as tf

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
from PIL import Image, ImageOps, ImageGrab

# face imports
from mtcnn.mtcnn import MTCNN
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder

# Define a flask app
app = Flask(__name__)
# Model saved with Keras model.save()
MODEL_PATH = 'models/fingerprint_model.h5'

# Load your trained model
model_finger = load_model(MODEL_PATH)
model_face = load_model('models/facenet_keras.h5')
loaded_model = load_model('models/facemodel.joblib')



# model._make_predict_function()
print('Models loaded. Start serving...')
print('Models loaded. Check http://127.0.0.1:5000/')


def fingerprint_predict(image_data, model_finger):
    img = cv2.imread(image_data)
    new_arr = cv2.resize(img, (224, 224))
    new_arr = np.array(new_arr / 255)
    new_arr = new_arr.reshape(-1, 224, 224, 3)
    preds = model_finger.predict(new_arr)
    return preds

def fingerprint_analysis():
    # Get the file from post request
    f = request.files['fingerprint_file']
    if not f:
        return None
    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', f.filename)
    f.save(file_path)
    # Make prediction
    preds = fingerprint_predict(file_path, model_finger)
    # Process your result for human
    pred_class = preds.argmax()  # Simple argmax
    CATEGORIES = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']
    score = tf.nn.softmax(preds[0])
    index = np.argmax(preds[0])
    predicted_class = CATEGORIES[index]
    probability = preds[0][index] * 100
    print("This fingerprint belongs to ", predicted_class, " with a ", probability, " percent confidence.")
    return [CATEGORIES[pred_class], probability, f]

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
def face_recognize(image):
    out_encoder = LabelEncoder()
    loaded_encoder = out_encoder
    loaded_predictor = loaded_model
    loaded_facenet = model_face
    CATEGORIES = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']
    loaded_encoder.fit(CATEGORIES)
    ## Extract the face and bounding box dimensions from the image by using pretrained MTCNN model
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
def face_predict(image_data, model_face):
    image = Image.open(image_data)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    face_recognize(pixels)



def face_analysis():
    # Get the file from post request
    f = request.files['face_file']
    if not f:
        return None
    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', f.filename)
    f.save(file_path)
    # Make prediction
    preds = face_predict(file_path, model_face)
    # Process your result for human
    pred_class = preds.argmax()  # Simple argmax
    CATEGORIES = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']
    score = tf.nn.softmax(preds[0])
    index = np.argmax(preds[0])
    predicted_class = CATEGORIES[index]
    probability = preds[0][index] * 100
    print("This face belongs to ", predicted_class +".")
    return [CATEGORIES[pred_class], probability, f]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        fingerprint_result = fingerprint_analysis()
        face_result = face_analysis()
        if not face_result or not fingerprint_result:
            print("Insufficient Data")
            return ("Insufficient Data. Please Try Again.")
        if face_result[0]==fingerprint_result[0]:
            print("Access Granted")
            if face_result[0]=='madonna':
                person='Madonna'
            else:
                person_name=face_result[0].split('_')
                person = str(person_name[0].title()) +" "+ str(person_name[1].title())
            return ("Access Granted. Welcome " + person +"!!")
        else:
            print("Access Denied")
            return ("Oops!! Access Denied")
    return None


if __name__ == '__main__':
    app.run(debug=True)
