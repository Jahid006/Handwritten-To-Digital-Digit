import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import Image

import numpy as np
from recognizer.model import get_model
from recognizer.train import compile_model
from config import PRETRAINED_DIR
model_ = get_model()
model_ = compile_model(model_)
model_.load_weights(PRETRAINED_DIR)

st.set_page_config("Digit Playground")
st.title("🎮 Digit Playground")
title_img = Image.open("1.png")

st.image(title_img)

st.header("🖼️ Digit Reconstruction", "recon")
img = None


with st.form("reconstruction"):
    recon_canvas = st_canvas(
        fill_color="rgb(0, 0, 0)",
        stroke_width=12,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="recon_canvas",
    )
    submit = st.form_submit_button("Convert to Digital Digit")
    if submit:
        img = Image.fromarray(recon_canvas.image_data,'RGBA').convert('L').resize((28,28))
        img = np.array(img)/255

        image_files = np.array(img).reshape(1,28,28,1)
        prediction = model_.predict(image_files)[0]

        img = prediction[0].reshape(28,28)*255
        img = Image.fromarray(img.astype('uint8')).convert('RGB').resize((150,150))
        st.image(img)
        
        confirm = st.form_submit_button("New Canvas?")
        if confirm: 
            recon_canvas.clear()
            



