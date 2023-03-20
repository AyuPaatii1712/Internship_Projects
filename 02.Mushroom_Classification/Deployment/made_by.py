import streamlit as st
st.header('**Made By**')
st.write('---')
col1, col2, col3= st.columns([3,6,4])


with col1:
    st.subheader("Name")
    st.write('Ayush Patidar')
    st.write('Anup Vetal')
    st.write('Farzan Nawaz')
    st.write('Prashant Khandekar')
    st.write('Prasad Waje')

with col2:
    st.subheader("Mail")
    st.write('ayushpatidar1712@gmail.com')
    st.write('anupsv1997@gmail.com')
    st.write('farzannawaz4787@gmail.com')
    st.write('pkhandekar108@gmail.com')
    st.write('prasadwaje2029@gmail.com')

with col3:
    st.subheader("Mob. No.")
    st.write('9131985346')
    st.write('8668314822')
    st.write('7898480467')
    st.write('7030870449')
    st.write('8999714455')
