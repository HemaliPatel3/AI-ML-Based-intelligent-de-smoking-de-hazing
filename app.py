from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from image_dehazer import image_dehazer
import base64

app = Flask(__name__)

# Initialize the dehazer
dehazer = image_dehazer(airlightEstimation_windowSze=15, boundaryConstraint_windowSze=3, C0=20, C1=300,
                        regularize_lambda=0.1, sigma=0.5, delta=0.85)

# Load haze detection model
haze_detection_model = load_model('C://Users//Hemali Patel//Desktop//sem 8 project//haze_detection_model1.h5')

# Function to check if the image is hazy
def is_hazy(image):
    # Preprocess the image for prediction
    image = cv2.resize(image, (128, 128))  # Adjusted size to match model's input shape
    image = np.expand_dims(image, axis=0)
    # Predict haze probability
    haze_prob = haze_detection_model.predict(image)
    # If haze_prob is above a threshold, consider it as hazy
    if haze_prob < 0.5:  # Adjust the threshold as needed
        return True
    else:
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dehaze', methods=['POST'])
def dehaze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Read the image
        img_np = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Check if the image is hazy
        if is_hazy(img):
            # Perform dehazing
            dehazed_img, _ = dehazer.remove_haze(img)

            # Encode dehazed image to base64 string
            _, buffer = cv2.imencode('.jpg', dehazed_img)
            dehazed_base64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({'result': dehazed_base64})
        else:
            return jsonify({'error': 'Image is clear and does not require dehazing'})

if __name__ == '__main__':
    app.run(debug=True)









'''from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from image_dehazer import image_dehazer
import base64

app = Flask(__name__)

# Initialize the dehazer
dehazer = image_dehazer(airlightEstimation_windowSze=15, boundaryConstraint_windowSze=3, C0=20, C1=300,
                        regularize_lambda=0.1, sigma=0.5, delta=0.85)

# Load haze detection model
haze_detection_model = load_model('C://Users//Hemali Patel//Desktop//sem 8 project//haze_detection_model1.h5')

# Function to check if the image is hazy
def is_hazy(image):
    # Preprocess the image for prediction
    image = cv2.resize(image, (128, 128))  # Adjusted size to match model's input shape
    image = np.expand_dims(image, axis=0)
    # Predict haze probability
    haze_prob = haze_detection_model.predict(image)
    # If haze_prob is above a threshold, consider it as hazy
    if haze_prob < 0.5:  # Adjust the threshold as needed
        return True
    else:
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dehaze', methods=['POST'])
def dehaze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Read the image
        img_np = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Check if the image is hazy
        if is_hazy(img):
            # Perform dehazing
            dehazed_img, _ = dehazer.remove_haze(img)

            # Encode dehazed image to base64 string
            _, buffer = cv2.imencode('.jpg', dehazed_img)
            dehazed_base64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({'result': dehazed_base64})
        else:
            return jsonify({'error': 'Image is clear'})

if __name__ == '__main__':
    app.run(debug=True)


'''












'''from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from image_dehazer import remove_haze
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dehaze', methods=['POST'])
def dehaze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Read the image
        img_np = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Perform dehazing
        dehazed_img, _ = remove_haze(img)

        # Encode dehazed image to base64 string
        _, buffer = cv2.imencode('.jpg', dehazed_img)
        dehazed_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'result': dehazed_base64})

if __name__ == '__main__':
    app.run(debug=True)
'''