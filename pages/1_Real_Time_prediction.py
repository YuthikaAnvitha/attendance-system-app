from home import st
from home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

st.set_page_config(page_title='Predictions')
st.subheader('Real Time Attendance')

#Retrieve the data from redis

with st.spinner('Retrieving Data from Redis db...'):
    redis_face_db=face_rec.retrieve_data(name='academy:register')
    st.dataframe(redis_face_db)
    
st.success('Data successfully retrieved from Redis')


#time
waitTime=30
setTime=time.time()
realtimepred=face_rec.RealTimePred()

#Real time prediction
#callback function

def video_frame_callback(frame):
    global setTime
    
    img = frame.to_ndarray(format="bgr24") #3d numpy array
    pred_img= realtimepred.face_prediction(img,redis_face_db,'facial_features',['Name','Role'],thresh=0.5)
    
    timenow=time.time()
    difftime=timenow-setTime
    if difftime>= waitTime:
        realtimepred.saveLogs_redis()
        setTime=time.time()
        
        print('Save data to redis database')
        
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="Realtimeprediction ", video_frame_callback=video_frame_callback,rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
