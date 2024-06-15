from flask import Flask, request, render_template
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from numpy import *
import sys
import keras
import logging



#for networks
from twilio.rest import Client

#EMAIL

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials


# Import the logging module
#logging.getLogger('tensorflow').setLevel(logging.INFO)  # Set TensorFlow logger level to INFO

import time
from gevent.pywsgi import WSGIServer
app = Flask('__name__')


# Define a dictionary of custom objects for loading models
custom_objects = {
    'KerasLayer': hub.KerasLayer
}

# Load the models with the custom_objects parameter
model_security = load_model('security_system_human.keras', custom_objects=custom_objects)
model_owner = load_model('owner classification.keras', custom_objects=custom_objects)

#model_security=load_model('security_system_human.h5')
#model_owner=load_model('owner classification.h5')


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/start',methods=['GET', 'POST'])
def index():
    return render_template('index1.html')


@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        # Load pre-trained face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Set a start time to track elapsed time
        start_time = time.time()

        captured = False
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error capturing frame.")
                break

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

            # Draw rectangles around detected faces for debugging
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the frame with detected faces
            cv2.imshow('Face Detection', frame)

            # If a face is detected and 2 seconds have passed, capture and save the image
            if len(faces) > 0 and time.time() - start_time >= 5:
                x, y, w, h = faces[0]
                face = frame[y:y+h, x:x+w]

                # Create the directory if it doesn't exist
                output_dir = "Cap_images"
                os.makedirs(output_dir, exist_ok=True)

                # Generate a unique image filename (you can modify this as needed)
                image_filename = os.path.join(output_dir, "captured_face.jpg")

                # Save the captured face image
                cv2.imwrite(image_filename, face)

                # Display a message and break the loop
                print("Face captured and saved.")
                captured = True
                break

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Break the loop if 2 seconds have passed without detecting a face
            if time.time() - start_time >= 5:
                print("No face detected within 5 seconds.")
                captured = False
                break

        # Release the webcam and close all windows
        cap.release()
        cv2.destroyAllWindows()

        if(captured==True):
        
            img_p = os.path.join('Cap_images', 'captured_face.jpg')
                   
            img = plt.imread(img_p)
            resize = tf.image.resize(img, (224, 224))
            resize = resize / 255.0
            img2 = expand_dims(resize, 0)
            
            values=model_security.predict(img2)

            #print("hai",values)
            print("##########################")
            #return str(int(values[0][0]))+"surya"
            #print(model_security.summary())

            #return "suya"

            value=values[0]
            if(value[0]>0.5):

                msg = "No Valid Human Is Detected , Please Try To Stay Properly."
                return render_template('output1.html',n=msg)
            else:
                return render_template('owner.html')
        
                        
        else:
            msg= 'No face detected within 5 seconds.'
            return render_template('output1.html',n=msg)
            
    return 'Method not allowed'


def drive_link_generator():
    
    # Authenticate with the Google Drive API
    gauth = GoogleAuth()
    scope = ['https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('E:\My work\Final_year\Bank-survalince-system-cd00abeee98f.json', scope)
    gauth.credentials = creds
    drive = GoogleDrive(gauth)


    # After saving the image

    image_name = "captured_face.jpg"  # Name of the saved image file

    # Upload the image to Google Drive
    file = drive.CreateFile({'title': image_name})
    
    file.SetContentFile(os.path.join('Cap_images', 'captured_face.jpg'))
        
    file.Upload()
    # Set the file's sharing permissions to make it publicly accessible
    file.InsertPermission({
        'type': 'anyone',
        'value': 'anyone',
        'role': 'reader',
    })

    # Get the link to the uploaded image
    image_link = file['alternateLink']
    print("Uploaded image link:", image_link)

    return image_link


def alert_send_pic(link):
    
    # Replace these variables with your Twilio account credentials
    account_sid = 'ACa3e6ca7b41a20e0ab3749c5b686b49e7'
    auth_token = 'cc6e5b2fdc06c18531fe0f34323fe27a'
    twilio_phone_number = '+16319002658'
    recipient_phone_number = '+919346303379'

    client = Client(account_sid, auth_token)

    # URL you want to send as a text message
    url = link
    #'https://drive.google.com/file/d/15gV8kLzTnzRVxZprklTxR0dnJfAr2amV/view?usp=drivesdk'

    # Message body containing the URL
    message_body = f'Check out this link, Today Your Account Has Been Accessed: {url}'

    # Send the message
    message = client.messages.create(
        body=message_body,
        from_=twilio_phone_number,
        to=recipient_phone_number
    )

    return "Your Locker Is Successfully Opened."

def alert_msg():    
    # Replace these variables with your Twilio account credentials
    account_sid = 'ACa3e6ca7b41a20e0ab3749c5b686b49e7'
    auth_token = 'cc6e5b2fdc06c18531fe0f34323fe27a'
    twilio_phone_number = '+16319002658'
    recipient_phone_number = '+919346303379'

    client = Client(account_sid, auth_token)

    # URL you want to send as a text message
    #url = link
    #'https://drive.google.com/file/d/15gV8kLzTnzRVxZprklTxR0dnJfAr2amV/view?usp=drivesdk'

    url="Dear,customer we have observered that your account has been tried by someone to open.So please kindly report to authorized people."

    # Message body containing the URL
    message_body = f'Check out this Notification: {url}'

    # Send the message
    message = client.messages.create(
        body=message_body,
        from_=twilio_phone_number,
        to=recipient_phone_number
    )

    return "You Are Not A Valid Owner To Access The Account, We Are Intimating To Owner"

def more_access(link):
    # Replace these variables with your Twilio account credentials
    account_sid = 'ACa3e6ca7b41a20e0ab3749c5b686b49e7'
    auth_token = 'cc6e5b2fdc06c18531fe0f34323fe27a'
    twilio_phone_number = '+16319002658'
    recipient_phone_number = '+919346303379'

    client = Client(account_sid, auth_token)

    # URL you want to send as a text message
    url = link
    #'https://drive.google.com/file/d/15gV8kLzTnzRVxZprklTxR0dnJfAr2amV/view?usp=drivesdk'

    #url="Dear,customer we have observered that your account has been tried someone to open.So please kindly report to authorized people."

    #Message body containing the URL
    message_body = f'Dear user today this person made more than 3 attempts to open your locker.So please kindly report to authority.Click here: {url}'

    # Send the message
    message = client.messages.create(
        body=message_body,
        from_=twilio_phone_number,
        to=recipient_phone_number
    )

    return "You Are Trying To Open Locker, We Transfered Your details To Owner."



@app.route('/owner', methods=['GET', 'POST'])
def validate_owner():
    
    img_p = os.path.join('Cap_images', 'captured_face.jpg')
            
                
    img = plt.imread(img_p)
    resize = tf.image.resize(img, (256, 256))
    resize = resize / 255.0
    img2 = expand_dims(resize, 0)
        
    values=model_owner.predict(img2)
    value=values[0]

    if(value[0]>0.5):
        return render_template('password.html')
    else:
        msg = alert_msg()
        return render_template('output1.html',n=msg)


attempts=0

@app.route('/validate', methods=['GET', 'POST'])
def validate1():
    
    
    if(request.method == "POST"):
        password = request.form['hiddenInput']

            


        if(password[15:]=="SURYA123@"):
            #call for link
            im_link=drive_link_generator()
            #call for alert
            notify=alert_send_pic(im_link)

            return render_template('output2.html',n=notify)

        else:
            global attempts
            attempts=attempts+1

            if(attempts<3):
                return render_template('password.html')
            else:
                
                #call for link
                im_link=drive_link_generator()
                #call for alert
                notify=more_access(im_link)
                return render_template('output1.html',n=notify)
                



if __name__ == "__main__":
    app.run(debug = False)
