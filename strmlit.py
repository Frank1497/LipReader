import streamlit as st
import imageio
import tempfile
import tensorflow as tf

from utils.data_prep import *
from utils.lipnet_model import *
from utils.face_mesh import FaceMesh




prep = Data_Prep()
lipnet = LipNet()
mesh = FaceMesh()

st.set_page_config(layout='wide')
st.image('https://caltechsites-prod.s3.amazonaws.com/scienceexchange/images/AI_HomePage-Teaser-Image-WEB.2e16d0ba.fill-1600x500-c100.jpg')
st.title('LIP READING APP')


#Adding option for videos
upload_vids = st.file_uploader("Choose a video file", type=['mp4', 'mpg'] )

col1, col2 = st.columns(2)
if upload_vids:
    with col1:
      st.text('col1')
      st.video(upload_vids)





    with col2:
      st.write('What the AI will see')
      tfile = tempfile.NamedTemporaryFile(delete=False)
      tfile.write(upload_vids.read())
      
      frames = prep.vid_frames(tfile.name)

      imageio.mimsave('test.gif', frames, fps=10)
      st.image('test.gif', width=400)
      
      st.write('Predictions')
      model = lipnet.build()
      model.load_weights('S:\\PROJECTS\\Lip_Reading\\checkpoints\\model_20.h5')
      yhat = model.predict(tf.expand_dims(frames, axis=0))
      st.write(f'{prep.tf_decoder(yhat)}')