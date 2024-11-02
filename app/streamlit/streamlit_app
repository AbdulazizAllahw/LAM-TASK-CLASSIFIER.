import os
import sys

# Add the parent directory of 'app' to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import streamlit as st
from app.chat_interface import chat_interface

st.set_page_config(page_title="LAM Task Classifier", page_icon="🤖", layout="wide")

chat_interface()
