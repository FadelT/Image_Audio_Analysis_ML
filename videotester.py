import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
import time

# load model
#model = load_model("best_model.h5")
#model=load_model('model_v6_23.hdf5')


#face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#face_haar_cascade=cv2.CascadeClassifier('C:/Users/user/Desktop/LastOne/haarcascade_frontalface_default.xml')
face_haar_cascade=cv2.CascadeClassifier('/tmp/haarcascade_frontalface_default.xml')


def analyze(img):

    #cap = cv2.VideoCapture(0)


    #ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    #if not ret:
        #   continue
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("img5.jpg", gray_img)
    
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.05, 5)
    print('faceeee')
    print(faces_detected)
    #print(faces_detected.shape)
    if isinstance(faces_detected,tuple)==False:
        if faces_detected.shape[0]>1:
            faces_detected=faces_detected[:1]
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (224, 224))
            #roi_gray.save('grayimg.jpg')
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            print('ok')
            try:
                obj = DeepFace.analyze(img_path = "img5.jpg", actions = ['age', 'gender', 'race', 'emotion'])
            except :
                cv2.putText(img, 'Face', (int(x), int(y)), cv2.FONT_HERSHEY_TRIPLEX , 1, (0, 0, 255), 2)
                resized_img = cv2.resize(img, (1000, 700))
                cv2.imwrite('imgout.jpg',img)
                resized_img = cv2.resize(img, (1000, 700))
                return img, 1, 0
            print("fdfd{}")
            print(obj)
            max_index=np.argmax(list(obj['emotion'].values()))
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            print(predicted_emotion)
            cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_TRIPLEX , 1, (0, 0, 255), 2)
            #cv2.imwrite('imgout.jpg',img)
            
        resized_img = cv2.resize(img, (1000, 700))
        cv2.imwrite('imgout.jpg',img)
        resized_img = cv2.resize(img, (1000, 700))
        return resized_img, predicted_emotion,1
    else:
        return img, 0, 0

    #except:
    """
    predictions = model.predict(img_pixels)

    # find max indexed array
    max_index = np.argmax(predictions[0])

    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    predicted_emotion = emotions[max_index]

    cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    """
    
        #print(resized_img)
        #cv2.imshow('Facial emotion analysis ', resized_img)

        #if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        #    break

    #cap.release()
    #cv2.destroyAllWindows
    #return resized_img
