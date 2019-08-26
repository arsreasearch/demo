from flask import Flask, render_template, Response, request , redirect, url_for
import cv2
import sys
import os
import sqlite3
from PIL import Image
import numpy
import datetime
import re, itertools
from flask import *  
from flask_sqlalchemy import SQLAlchemy
import shutil

project_dir = os.path.dirname(os.path.abspath(__file__))
database_file = "sqlite:///{}".format(os.path.join(project_dir, "User.db"))



app = Flask(__name__)


#########constants###############################
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
model = cv2.face.LBPHFaceRecognizer_create() 
trainingData = 'recognizer/trainingData.yml'
conn = sqlite3.connect('database.db' ,check_same_thread=False)
cur = conn.cursor()
#################################################


app.config["SQLALCHEMY_DATABASE_URI"] = database_file
db = SQLAlchemy(app) 
 
class User(db.Model):
    idnum = db.Column(db.Integer, unique=True, nullable=False, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    images = db.Column(db.String(80))
    status = db.Column(db.String(80))



    
# @app.route('/')
# def index():



def capture():
    
    face_cascade = cv2.CascadeClassifier(haar_file) 
    webcam = cv2.VideoCapture(0) 

    (images, lables, names, id) = ([], [], {}, 0) 
    for (subdirs, dirs, files) in os.walk(datasets): 
        for subdir in dirs: 
            names[id] = subdir 
            subjectpath = os.path.join(datasets, subdir) 
            for filename in os.listdir(subjectpath): 
                path = subjectpath + '/' + filename 
                lable = id
                images.append(cv2.imread(path, 0)) 
                lables.append(int(lable)) 
        id += 1

    
    model.read(trainingData)

    while True: 
        (_, im) = webcam.read() 
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
        for (x, y, w, h) in faces: 
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            face = gray[y:y + h, x:x + w]  
            # Try to recognize the face 
            prediction = model.predict(face) 
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3) 
            if prediction[1]<500: 
                cv2.putText(im, '% s - %.0f' % 
        (names[prediction[0]], prediction[1]), (x-10, y-10),  
        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
            else: 
                cv2.putText(im, 'not recognized',  
        (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
        
            cv2.imshow('OpenCV', im) 
            
            key = cv2.waitKey(10) 
            if key == 27: 
                break

    # return render_template('index.html')

@app.route('/register', methods =['GET', 'POST'])
def register():
    if request.method == 'POST':
        idnum = request.form['id']
        username = request.form['username']
        add_employee(idnum, username)
        return render_template('register.html')
    return render_template('register.html')

def add_employee(idnum, username):
    me = User(idnum=idnum, name = username)
    db.session.add(me)

    datasets = 'datasets'  
    sub_data = username     
    
    path = os.path.join(datasets, sub_data) 
    if not os.path.isdir(path): 
        os.mkdir(path) 
        
    face_cascade = cv2.CascadeClassifier(haar_file) 
    webcam = cv2.VideoCapture(0)  
    
    count = 1
    while count < 100:  
        (_, im) = webcam.read() 
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
        for (x, y, w, h) in faces: 
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            face = gray[y:y + h, x:x + w] 
            src = str('% s/% s.png' % (path, count))
            cv2.imwrite('% s/% s.png' % (path, count), face)
        count += 1
        
        cv2.imshow('OpenCV', im) 
        key = cv2.waitKey(10) 
        if key == 27: 
            break

    db.session.commit()
    


@app.route('/train')
def train():
    row = User.query.all()
    return render_template("train.html", row=row)


@app.route('/train_employee')
def train_employee():
    
    if not os.path.exists('./recognizer'):
        os.makedirs('./recognizer')
     
    (images, lables, names, id) = ([], [], {}, 0) 
    for (subdirs, dirs, files) in os.walk(datasets): 
        for subdir in dirs: 
            names[id] = subdir 
            subjectpath = os.path.join(datasets, subdir) 
            for filename in os.listdir(subjectpath): 
                path = subjectpath + '/' + filename 
                lable = id
                images.append(cv2.imread(path, 0)) 
                lables.append(int(lable)) 
            id += 1
    
    (images, lables) = [numpy.array(lis) for lis in [images, lables]] 
    
 
    model.train(images, lables) 
    model.save(trainingData)
    return redirect('train')

@app.route("/delete", methods=["POST"])
def delete():
    title = request.form.get("title")
    username = User.query.filter_by(name=title).first()
    db.session.delete(username)
    db.session.commit()
    shutil.rmtree(username)
    return redirect("/gallery")


#######################################

if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True)
