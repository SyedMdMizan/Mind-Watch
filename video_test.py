import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
import pickle
import neattext.functions as nfx
from keras.preprocessing.sequence import pad_sequences
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt

# load model
model = load_model("model_filter.h5")
text_model = load_model("model.h5")

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def clean_text(text):
    cleaned_text = []
    for sent in text:
        sent = sent.lower()
        sent = nfx.remove_special_characters(sent)
        sent = nfx.remove_stopwords(sent)
        cleaned_text.append(sent)
    return cleaned_text

def predict_text_depression(text):
    cleaned_text = clean_text([text])
    text_seq = tokenizer.texts_to_sequences(cleaned_text)
    text_pad = pad_sequences(text_seq, maxlen=40)
    prediction = text_model.predict(text_pad)[0][0]
    return prediction > 0.5, prediction


cap = cv2.VideoCapture(0)
post = input("Please enter you feelings: ")
is_depressed, confidence = predict_text_depression(post)
flag = False
if is_depressed:
    flag = True

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        

    resized_img = cv2.resize(test_img, (1000, 700))
    if flag is True:
        cv2.imshow('Potential Dpression Detected ', resized_img)
    else:
        cv2.imshow('Not Dpressed', resized_img)


    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows