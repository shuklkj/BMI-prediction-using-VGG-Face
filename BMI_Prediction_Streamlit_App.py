#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import streamlit as st
from mtcnn import MTCNN
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as img


# In[2]:


# Load the pre-trained BMI model
# set the path to the directory containing the images
path = "/Users/kajalshukla/Desktop/Quarter-3/ML & Predictive Analytics/ML_Final_Project/Final_Project"
model_path = os.path.join(path, 'bmi_pred_model_v10.h5')
bmi_model = load_model(model_path)


# In[3]:


# Define the face detector
detector = MTCNN()


# In[4]:


def run_realtime_camera():
    # Create a video capture object
    cap = cv2.VideoCapture(0)

    bmi_values = []

    # Create a Streamlit placeholder for the video
    video_placeholder = st.empty()

    # Start the video capture loop
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Detect faces in the frame
        faces = detector.detect_faces(frame)

        # Iterate over the detected faces
        for face in faces:
            # Extract the face coordinates
            x, y, width, height = face['box']
            x1, y1 = x + width, y + height

            # Preprocess the face image
            face_img = frame[y:y1, x:x1]
            face_img = cv2.resize(face_img, (224, 224))
            face_img = np.expand_dims(face_img, axis=0)
            face_img = np.array(face_img) / 255.0 

            # Get the predicted BMI value (assuming you have a regression model)
            predicted_bmi = bmi_model.predict(face_img)
            bmi_values.append(predicted_bmi[0][0])

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

            # Display the predicted BMI value on the frame if confidence is above threshold
            cv2.putText(frame, f'BMI: {bmi_values[-1]:.2f}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame in Streamlit
        video_placeholder.image(frame, channels='BGR')

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()


# In[5]:


def upload_image():
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = np.array(image) / 255.0
        
        predicted_bmi = bmi_model.predict(image)
        
        st.header("Predicted BMI:")
        st.title(predicted_bmi[0][0])       
        


# In[7]:


def main():
    st.title("BMI Prediction App")
    st.sidebar.title("Options")
    
    option = st.sidebar.selectbox("Select an option", ("Upload Image", "Capture from Camera"))
    
    if option == "Capture from Camera":
        run_realtime_camera()
    elif option == "Upload Image":
        upload_image()
    else:
        st.write("")


# In[8]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




