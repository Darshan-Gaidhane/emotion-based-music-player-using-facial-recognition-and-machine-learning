
import cv2
import numpy as np
from keras.models import load_model
import sys
import tensorflow as tf

import webbrowser
key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)

while True:
    try:
        check, frame = webcam.read()
     
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):
           
            webcam.release()
        
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

      
emotion_classifier = load_model('keras_model/model_5-49-0.62.hdf5', compile=False)
faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')
model = load_model('keras_model/model_5-49-0.62.hdf5')

def test_image(addr):
    target = ['angry','disgust','fear','happy','sad','surprise','neutral']
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    im = cv2.imread(addr)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1)
    
    
    for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2,5)
            face_crop = im[y:y+h,x:x+w]
            face_crop = cv2.resize(face_crop,(48,48))
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_crop = face_crop.astype('float32')/255
            face_crop = np.asarray(face_crop)
            face_crop = face_crop.reshape(1, 1,face_crop.shape[0],face_crop.shape[1])
            result = target[np.argmax(model.predict(face_crop))]
            
            cv2.putText(im,result,(x,y), font, 1, (200,0,0), 3, cv2.LINE_AA)
    preds = emotion_classifier.predict(face_crop)[0]
    emotion_probability = np.max(preds)
    label = target[preds.argmax()]
   
   
 
    for (i, (emotion, prob)) in enumerate(zip(target, preds)):
                
                text = "{}: {:.2f}%".format(emotion, prob * 100)

               
                print(text)
    
            
    cv2.imshow('result', im)
    cv2.imwrite('new.jpg',im)
    cv2.waitKey(0) 
    
               
    if(result=='angry'):
        print("ANGRY")
        webbrowser.open('anger.html')
    elif(result=='disgust'):
        print("DISGUST")
        webbrowser.open('disgust.html')
    elif(result=='scared'):
        print("SACARED")
        webbrowser.open('scared.html')
    elif(result=='happy'):
        print("HAPPY")
        webbrowser.open('happy.html')
    elif(result=='sad'):
        print("SAD")
        webbrowser.open('sad.html')
    elif(result=='surprised'):
        print("SURPRISED")
        webbrowser.open('surprised.html')
    elif(result=="neutral"):
        print("NEUTRAL")
        webbrowser.open('neutral.html')                     


    
if __name__=='__main__':
    image_addres = 'saved_img.jpg'
    test_image(image_addres)
    