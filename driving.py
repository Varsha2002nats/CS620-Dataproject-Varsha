# Import necessary libraries
import socketio  # For WebSocket communication between the simulator and server
import eventlet  # For asynchronous server
import numpy as np  # For numerical operations
from flask import Flask  # For creating the Flask application
from keras.models import load_model  # To load the trained machine learning model
import base64  # For decoding base64 image data
from io import BytesIO  # To handle image input as byte streams
from PIL import Image  # For image manipulation
import cv2  # For computer vision operations

# Create a Socket.IO server to handle real-time communication
sio = socketio.Server()

# Initialize Flask application
app = Flask(__name__)  # Flask is used as the web framework
speed_limit = 10  # Set the speed limit for the vehicle

# Function to preprocess the input image
def img_preprocess(img):
    # Crop the image to remove unnecessary parts (sky and car hood)
    img = img[60:135, :, :]
    # Convert the image to YUV color space, which is often used in car driving models
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Apply Gaussian Blur to smoothen the image and reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Resize the image to the input size expected by the model
    img = cv2.resize(img, (200, 66))
    # Normalize the image pixel values to the range [0, 1]
    img = img / 255
    return img

# Event handler for telemetry data sent from the simulator
@sio.on('telemetry')
def telemetry(sid, data):
    # Retrieve the current speed of the car
    speed = float(data['speed'])
    # Decode the incoming image data from base64 format
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    # Convert the image to a NumPy array for processing
    image = np.asarray(image)
    # Preprocess the image to prepare it for the model
    image = img_preprocess(image)
    # Expand dimensions to match the input shape required by the model
    image = np.array([image])
    # Predict the steering angle using the pre-trained model
    steering_angle = float(model.predict(image))
    # Calculate the throttle based on the speed (reduce throttle as speed approaches the limit)
    throttle = 1.0 - speed / speed_limit
    # Print the steering angle, throttle, and speed for debugging
    print('{} {} {}'.format(steering_angle, throttle, speed))
    # Send the steering and throttle commands back to the simulator
    send_control(steering_angle, throttle)

# Event handler for when the simulator connects to the server
@sio.on('connect')
def connect(sid, environ):
    print('Connected')  # Log the connection event
    # Send initial control commands (neutral steering and no throttle)
    send_control(0, 0)

# Function to send control commands to the simulator
def send_control(steering_angle, throttle):
    # Emit the 'steer' event with the calculated steering angle and throttle
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),  # Convert to string for transmission
        'throttle': throttle.__str__()  # Convert to string for transmission
    })

# Main entry point of the program
if __name__ == '__main__':
    # Load the trained model for predicting steering angles
    model = load_model('D:\\self drive simulator h5 files\\model\\model.h5')
    # Wrap the Flask app with the Socket.IO middleware
    app = socketio.Middleware(sio, app)
    # Start the server to listen on port 4567 and handle incoming simulator requests
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
