import cv2
import streamlit as st
import numpy as np
from PIL import Image
import face_recognition
import mediapipe as mp

#Drawing and facemesh utilities
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
model_face_mesh = mp_face_mesh.FaceMesh()

#For setting background image
p_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
model = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

add_selectbox = st.sidebar.selectbox(
    "Seelct the operation to be performed",
    ("About", "Grayscale Conversion", "Face Meshing", "Face Recognition", "Set Background")
)

##################################################### ABOUT SECTION ############################################################
if add_selectbox == "About":
    st.title("Welcome!")
    st.write("This app is been designed to perform various OpenCV operations. The operations include:")
    st.write("1. Image conversion to grayscale")
    st.write("2. Face Meshing")
    st.write("3. Face Recognition using your webcam feed")
    st.write("4. Set an background image for with your webcam feed")
    st.write("Select an option from the sidebar")


###################################################### FACE RECOG SECTION #########################################################
elif add_selectbox == "Face Recognition":
    st.title("Face Recogniton using webcam feed")
    st.write("This application is trained to recognise Robert Downey Jr, Chris Evans and Saudhey(me). "
    "It computes the encodings of the known faces and the test faces, and compares them to find a match. "
    "The camera feed is taken as individual frames and the encodings are compared with each frame.")
    st.write("Check to run camera and click on start recognition")

    #Loads images and computes their encodings
    chris = face_recognition.load_image_file('files/chris.jpg')
    chris_encodings = face_recognition.face_encodings(chris)[0]

    robert = face_recognition.load_image_file('files/Robert.jpg')
    robert_encodings = face_recognition.face_encodings(robert)[0]

    saudhey = face_recognition.load_image_file('files/saudhey.jpg')
    saudhey_encodings = face_recognition.face_encodings(saudhey)[0]

    #Arrays of known face encodings and their encodings
    known_face_encodings = [chris_encodings,robert_encodings,saudhey_encodings]
    known_face_names = ["Chris Evans","Robert Downey Jr","Saudhey Burra"]

    recog = st.sidebar.button("Start Recognition")
    run = st.checkbox("Run Camera")
    
    while run:
        FRAME_WINDOW = st.image([])         
        cap = cv2.VideoCapture(0)        #Gets a refernce to webcam 0

        if recog:
            flag, frame = cap.read()     #Reading frame by frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      #cnvts to RGB
            small_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)  #Resizing the image for faster processing

            #Extracting all the faces and their encodings for a single frame
            face_locations = face_recognition.face_locations(small_frame)              
            face_encodings = face_recognition.face_encodings(small_frame, face_locations) 

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding) #Compares and checks if face is a match for known images
                name = "Unknown"                                                              #Deafult
                #uses the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                #if a match is found, the the faace name array is used to name the face.
                if matches[best_match_index]:                                   
                    name = known_face_names[best_match_index]  
                face_names.append(name)

            #To display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                #For the box around the faces
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                #For the label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            FRAME_WINDOW.image(frame)

        else:
            st.write('Camera not found')    

############################################## GRAYSCALE SECTION################################################################
elif add_selectbox == "Grayscale Conversion":
    image_file_path = st.sidebar.file_uploader("Upload an image")
    st.title("Grayscale Conversion")
    st.write("Upload an image to change it to graysale")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)

    cnvt = st.sidebar.button("Convert")
    if cnvt:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        st.image(gray_image)
        st.write("Successfully Coverted to grayscale")

##################################################FACE MESHING################################################
elif add_selectbox == "Face Meshing":
    image_file_path = st.sidebar.file_uploader("Upload image")
    st.title("Face Meshing")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)

    mesh = st.sidebar.button("Constuct Mesh")
    if mesh:
        results = model_face_mesh.process(image)
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, face_landmarks)
        st.image(image)

####################################### BACKGROUND ##################################################
elif add_selectbox == "Set Background":
    image_file_path = st.sidebar.file_uploader("Upload background")
    st.title("Set a background")
    if image_file_path is not None:
        bg_image = np.array(Image.open(image_file_path))
        st.sidebar.image(bg_image)     
    bgset = st.sidebar.button("Set Background")
    if bgset:
        FRAME_WINDOW = st.image([])         
        cap = cv2.VideoCapture(0)        #Gets a refernce to webcam 0

        while cap.isOpened():
            flag, frame = cap.read()
            if not flag:
                print("Cant capture video")                        #Reading frame by frame
                break
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      #cnvts to RGB
                result = model.process(frame)    
                condition = np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.1 		#Stacks the image vertically
                bg_image = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
                output_image = np.where(condition, frame, bg_image)
            FRAME_WINDOW.image(output_image)        
