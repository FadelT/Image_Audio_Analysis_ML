# streamlit_audio_recorder by stefanrmmr (rs. analytics) - version April 2022

import os
from matplotlib.cbook import contiguous_regions
import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from test_model import process_image
from PIL import Image
import matplotlib as plt
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
from videotester import analyze
import tensorflow as tf
from streamlit import caching
from audio_listener import takeCommand, get_audio_classification
from gtts import gTTS

curdir=os.path.abspath(os.getcwd())
curdir = '/'.join(curdir.split('\\'))

api_key='e4d13c610fff4893926ae557b647d99f'



os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
gpu_options = tf.GPUOptions(allow_growth=True) 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))
filename ="/tmp/microphone-results.wav"
api_key=api_key

def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return audio_bytes

def audio_analysis():
    st.title('streamlit audio analysis with Machine Learning')
    st.header("1. Record your own voice")

    txt_filename = st.text_input("Choose a filename: ")
    filename="microphone-results.wav"
    button=st.button(f"Click to Record")

    if button:
        
        if txt_filename == "":
            st.warning("Choose a filename.")
        else:
            #speech = gTTS(text=txt_filename)
            #speech.save('speech.wav')
            st.write('Wait')
            takeCommand()
            content_classification,confidence_of_prediction=get_audio_classification(api_key=api_key,filename=filename)
            st.audio(read_audio(filename))
            if content_classification!=0 and confidence_of_prediction !=0:
                st.write("I am {} sure that this audio is about {}".format(confidence_of_prediction*100,content_classification))
            else:
                st.write("Oh My God, I haven't understood. Can you try again? Make sure it is Shakespeare's Language (English)")
                st.markdown(":confused:")


    pass

def text_analysis():
    st.title('streamlit Text analysis with Machine Learning')
    st.header("1. Write your text here")

    txt = st.text_input("You can write here, just type in! ")
    filename="text_to_speech.wav"
    button=st.button(f"Click to Analyse your text")

    if button:
        
        if txt == "":
            st.warning("Oh you forgot to write your text, please type in and retry!")
        else:
            speech = gTTS(text=txt,lang='en')
            speech.save("{}/{}".format(curdir,filename))
            st.write('Wait')
            content_classification,confidence_of_prediction=get_audio_classification(api_key=api_key,filename="{}/{}".format(curdir,filename))
            st.audio(read_audio("{}/{}".format(curdir,filename)))
            if content_classification!=0 and confidence_of_prediction !=0:
                st.write("I am {} sure that this audio is about {}".format(confidence_of_prediction*100,content_classification))
            else:
                st.write("Oh My God, I haven't understood. Can you try again? Make sure it is Shakespeare's Language (English)")
                st.markdown(":confused:")

    pass

def audiorec_demo_app():

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    # Custom REACT-based component for recording client audio in browser
    build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
    # specify directory and initialize st_audiorec object functionality
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)

    # TITLE and Creator information
    st.title('Streamlit Image Analysis with Machine Learning')
    
    #st_audiorec()
    #st.audio('upload a wav audio file')
    file=st.file_uploader('upload a file',type=['jpg'])
    
    if file :
        #print(type(file))
        image = Image.open(file)
        st.image(image)
        image.save('{}/img4.jpg'.format(curdir))
        img = cv2.imread('{}/img4.jpg'.format(curdir))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rclasses, rscores, rbboxes = process_image(img)
        print(rclasses)
        print(type(rclasses))
        print('sjddi')
        imgg,result,c=analyze(img)
        if result==0:
            st.write('Apparently, there is no human face !')
            if rclasses.shape[0]>0:
                st.write('But It seems that there is dangerous thing such as Knive or Gun or Blood in it. I am not sure, I am still learning')
        elif isinstance(result,str) and c==1:
            st.write('This person looks {}'.format(result))
            img=cv2.imread('{}/imgout.jpg'.format(curdir))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img)
            if rclasses.shape[0]>0:
                st.write('It seems that there is dangerous thing such as Knive or Gun or Blood in it. I am not sure, I am still learning')
        elif result==1 and c==0:
            st.write('{}'.format(rclasses))
            if rclasses.shape[0]==0:
              st.write("There's a human face but I don't know his emotions")
              img=cv2.imread('{}/imgout.jpg'.format(curdir))
              img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              st.image(img)
            elif rclasses.shape[0]>0:
              st.write("There's a human face but I don't know his emotions. More, there's a dangerous thing like  Knive or Gun or Blood in it!")
              img=cv2.imread('{}/imgout.jpg'.format(curdir))
              img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              st.image(img)
        print('Ok')
        

def start_job() :  
    # DESIGN implement changes to the standard streamlit UI/UX
    st.set_page_config(page_title="streamlit_audio_recorder")
    # Design move app further up and remove top padding
    st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
        unsafe_allow_html=True)
    # Design change st.Audio to fixed height of 45 pixels
    st.markdown('''<style>.stAudio {height: 45px;}</style>''',
        unsafe_allow_html=True)
    # Design change hyperlink href link color
    st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
        unsafe_allow_html=True)  # darkmode
    st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
        unsafe_allow_html=True)  # lightmode

    st.markdown('Implemented by '
        '[N Bouyaa KASSINGA](https://www.linkedin.com/in/n-bouyaa-kassinga-818a02169/) - '
        'view project source code on '
        '[GitHub](https://github.com/FadelT/Image_Audio_Analysis_ML)')
    st.write('\n\n')

    options = st.sidebar.selectbox('What task do you wish?', ('Image Analysis', 'Audio Analysis','Text Analysis'))
    if options=='Audio Analysis':
        audio_analysis()
    elif options=='Image Analysis':
        audiorec_demo_app()
    elif options=='Text Analysis':
        text_analysis()

    st.write('You selected {}'.format(options))

    pass


if __name__ == '__main__':
    
    
    # call main function
    start_job()