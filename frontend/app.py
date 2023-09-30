import streamlit as st
import requests

st.write("Welcome")
url = 'https://phoneme-service-wifbxua65a-ew.a.run.app'
request = requests.get(url).json()["greeting"]
st.write(request)
