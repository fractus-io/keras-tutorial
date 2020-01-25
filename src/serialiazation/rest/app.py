'''

 Copyright (c) 2019 Fractus IT d.o.o. <http://fractus.io>
 
'''

# importing the libraries 
from flask import Flask
import flask
from keras.applications import ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = None

def load_model():
    # load pre trained model ResNet50
    global model
    model = ResNet50(weights="imagenet")
    
def prepare_image(image, target):

    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
         
    # resize input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    
    # return the preprocessed image
    return image
    
@app.route("/predict", methods=["POST"])
def predict():
    
    data = {"success": False}
    
    if flask.request.method == "POST":
        if flask.request.files["image"]:
            print("image")
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            
            # prepare image and prepare for classification 
            image = prepare_image(image, target=(224, 224))
            data["predictions"] = []
            
            # classify the input image
            predicts = model.predict(image)  
            results = imagenet_utils.decode_predictions(predicts)
            
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True                
    
    return flask.jsonify(data)
    
        
@app.route('/')
def index():
    return flask.jsonify("Hello World")


if __name__ == '__main__':

    load_model()
    app.run(host='0.0.0.0', debug=False, threaded=False)