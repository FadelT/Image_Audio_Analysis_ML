import speech_recognition as sr
import requests
from time import sleep
import streamlit as st

def takeCommand():
    #st.write('start')
    #st.write('recording....')
    r = sr.Recognizer()
     
    with sr.Microphone() as source:
         
        st.write("Listening.......")
        r.pause_threshold = 1
        audio = r.listen(source,timeout=300)
        # write audio to a WAV file
        with open("/tmp/microphone-results.wav", "wb") as f:          
            f.write(audio.get_wav_data())
        st.write('record finished')
    
    pass

#takeCommand()



def read_file(filename, chunk_size=5242880):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(chunk_size)
            if not data:
                break
            yield data



def get_audio_classification(api_key,filename):
    headers = {'authorization': api_key}
    response = requests.post('https://api.assemblyai.com/v2/upload',
                            headers=headers,
                            data=read_file(filename))

    audio_url = response.json()['upload_url']

    endpoint = "https://api.assemblyai.com/v2/transcript"

    json = {
    "audio_url": audio_url,
    "content_safety": True
    }

    headers = {
        "authorization": api_key,
        "content-type": "application/json"
    }

    transcript_input_response = requests.post(endpoint, json=json, headers=headers)



    transcript_id = transcript_input_response.json()["id"]

    print('5. Extract transcript ID')

    # 6. Retrieve transcription results
    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    headers = {
        "authorization": api_key,
    }

    transcript_output_response = requests.get(endpoint, headers=headers)
    

    while transcript_output_response.json()['status'] != 'completed':
        sleep(5)
        print('Transcription is processing ...')
        transcript_output_response = requests.get(endpoint, headers=headers)
    text_from_audio=transcript_output_response.json()['text']
    print(transcript_output_response.json()['content_safety_labels'])
    if len(transcript_output_response.json()['content_safety_labels']['results'])!=0:
        content_classification=transcript_output_response.json()['content_safety_labels']['results'][0]['labels'][0]['label']
        confidence_of_prediction=transcript_output_response.json()['content_safety_labels']['results'][0]['labels'][0]['confidence']
    else:
        confidence_of_prediction=0
        content_classification=0
    return content_classification,confidence_of_prediction

#transcript_output_response=get_audio_classification(api_key,filename)
#text_from_audio=transcript_output_response.json()['text']
#content_classification=transcript_output_response.json()['content_safety_labels']['results'][0]['labels'][0]['label']
#confidence_of_prediction=content_classification=transcript_output_response.json()['content_safety_labels']['results'][0]['labels'][0]['confidence']

#print(content_classification)
