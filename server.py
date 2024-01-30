import os

from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import pickle

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import keras
# model = keras.models.load_model('./location.keras')
app = Flask(__name__)

# Load your trained model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Define the custom Adam optimizer class
class CustomAdam(Adam):
    pass

# Register the custom optimizer
custom_objects = {'CustomAdam': CustomAdam}

# Load your trained model
model = load_model('model/sq_mode.h5', custom_objects=custom_objects)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static', secure_filename(f.filename))
        f.save(file_path)
        print(f)
        fn = f'static/{f.filename}'
        # Make prediction
        # img = image.load_img(file_path, target_size=(16, 32))
        # x = image.img_to_array(img)
        # x = np.true_divide(x, 255)
        # x = np.expand_dims(x, axis=0)
        # Make prediction
        img = image.load_img(file_path, target_size=(28, 28), color_mode='grayscale')
        x = image.img_to_array(img)
        x = np.true_divide(x, 255)
        x = x.reshape(1, 784)

        preds = model.predict(x)
        deep_net_pred_class = np.argmax(preds, axis=-1)


        # train_predictions = regression_model.predict(train_images)
        # test_predictions = regression_model.predict(test_images)

        # Later you can load the model from disk
        loaded_model = pickle.load(open('model/LoggesticModel.sav', 'rb'))
        # preds = model.predict(x)
        # pred_class = np.argmax(preds, axis=-1)
        loggestic_pred_class = loaded_model.predict(x)

        # return str(pred_class)    
        fn=fn[7:]
        return render_template('prediction.html' ,  pred={"nnPred":deep_net_pred_class[0], "lg":loggestic_pred_class[0], "filePath":fn, })
    return None

@app.route('/static/<filename>')
def serve_image(filename):
    
    image_path = os.path.join(app.root_path, 'static', filename)
    return send_file(image_path, mimetype='image/jpeg')  # Adjust mimetype based on your image type


if __name__ == '__main__':
    app.run(port=5000, debug=True)
