import os
from flask import Flask, request, render_template, send_from_directory, jsonify, redirect, url_for
import cv2
import numpy as np 
import pandas as pd
import pickle
import joblib
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = 'static'

# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Memuat model dari file joblib
HKNeighborclassifier = joblib.load('classifier.joblib')
lda = joblib.load('lda_model.joblib')

@app.route('/')
def index():
    return render_template('index.html', css_file='css/app.css')

@app.route('/dataset')
def dataset():
    with open('static/2D-LDA.csv', 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)
    return render_template('dataset.html')

@app.route('/confusionmatrix')
def confusionmatrix():
    return render_template('confusionmatrix.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediksi')
def deteksi():
    return render_template('prediksi.html')

@app.route('/uploads/<filename>')
def display_img(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/convert', methods=['POST'])
def convert():

    def cek_keadaan(hasil):
        # classes = ['Normal', 'Moderate', 'Mild', 'Severe', 'Proliferative']
        # predicted_class = classes[np.argmax(hasil)]

        if hasil == 0:
            return "Mata Anda Normal, Jaga Selalu Kesehatan Ya!"
        if hasil == 1:
            return "Mata Anda Berpotensi Retinopati Diabetik dengan tingkat keparahan Mild."
        if hasil == 2:
            return "Mata Anda Berpotensi Retinopati Diabetik dengan tingkat keparahan Moderate."
        if hasil == 3:
            return "Mata Anda Berpotensi Retinopati Diabetik dengan tingkat keparahan Severe."
        if hasil == 4:
            return "Mata Anda Berpotensi Retinopati Diabetik dengan tingkat keparahan Proliveratife."
        else:
            return "Tidak Dikenali"

    if 'img' not in request.files:
        return "No img file provided"

    img_file = request.files['img']

    if img_file.filename == '':
        return "No Selected File"
    
    if img_file and allowed_file(img_file.filename):
        temp_img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img_file.save(temp_img_path)
        img = cv2.imread(temp_img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.medianBlur(img, 5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv2.resize(img, (100, 100))
        img = img / 255.0
        img_flat = img.flatten().reshape(1, -1)
        # feature_columns = ['LDA1','LDA2']
        # X = [feature_columns]
        # y = ['normal, mild, moderate, severe, proliferative']
        # lda = LinearDiscriminantAnalysis(n_components=2)
        X_lda = lda.transform(img_flat)
        df_lda = pd.DataFrame(X_lda, columns=['LDA1', 'LDA2'])
        HassanatPred = HKNeighborclassifier.predict(df_lda)
        keadaan = cek_keadaan(HassanatPred)

    return render_template('hasil.html', keadaan=keadaan)



if __name__ == '__main__':
    app.run(debug=True)