import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
from home import face_rec
import av


st.set_page_config(page_title='Registration form',layout='centered')
st.subheader('Registration form')


#init registration form
registration_form=face_rec.RegistrationForm()


#step1: collect person name and role
#form 
person_name=st.text_input(label='Name',placeholder='First and Last Name')
role= st.selectbox(label='Select your role',options=('Dayscholar','Teacher','Incharge'))

#step2: collect facial embedding of that person
def video_callback_func(frame):
    img = frame.to_ndarray(format="bgr24") #3d numpy array
    reg_img,embedding= registration_form.get_embedding(img)
    # two step process
    #1st step save data into local computer txt
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)
    
    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")


webrtc_streamer(key="Registration", video_frame_callback=video_callback_func,rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })


#step3: save the data in redis database


if st.button('Submit'):
    return_val=registration_form.save_data_in_redis_db(person_name,role)
    if return_val == True:
        st.success(f'{person_name} registered successfully')
        
    elif return_val == 'name_false':
        st.error('Please  enter the name: Name cannot be empty or spaces')
        
    elif return_val == 'file_false':
        st.error('face_embedding.txt is not found. Please refresh the page and execute it again. ')