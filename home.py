import streamlit as st


st.set_page_config(page_title='Attendance_system')
st.header('Attendance system')

with st.spinner('Loading Models and connecting to Redis db...'):
    import face_rec
    
st.success('Model loaded successfully')
st.success('Redis db successfully connected')
