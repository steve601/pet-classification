from flask import Flask,render_template,request
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model('artifacts\cat_dogmodel.keras')


UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route("/detect",methods = ['POST'])
def recognize():
    imgfile = request.files['digit']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imgfile.filename)
    imgfile.save(image_path)
    
    img = load_img(image_path)
    img = img.resize((32,32))
    img_arr = img_to_array(img)/ 255.0
    img_arr = img_arr.reshape(1, 32,32, 3)
    
    pred = model.predict(img_arr)[0,0]
    
    msg = 'This is a cat' if pred < 0.5 else 'This is a dog'
    
    return render_template('index.html',text=msg,img_path = image_path)

if __name__ == '__main__':
    app.run(debug=True)