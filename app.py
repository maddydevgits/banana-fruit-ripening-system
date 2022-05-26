import streamlit as st
import os
from PIL import Image, ImageOps
import numpy as np
import tensorflow.keras
import random

labels={
    0: 'Too Early',
    1: 'Rippen',
    2: 'Too Late'
}

st.title('Fruit Ripening System')
model=tensorflow.keras.models.load_model('keras_model.h5')

def load_image(image_File):
    img=Image.open(image_File)
    return img

st.subheader('Upload Input Image')
src_image_file=st.file_uploader('upload images',type=['png','jpg','jpeg'])

if src_image_file is not None:
    file_details={"filename":src_image_file,"filetype":src_image_file.type,"filesize":src_image_file.size}
    st.write(file_details)
    st.image(load_image(src_image_file),width=250)

    with open(os.path.join("uploads","src.jpg"),"wb") as f:
        f.write(src_image_file.getbuffer())
        st.success("File Saved")

    data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
    #open the image
    image=Image.open('uploads/src.jpg') # test.jpg
    # resize the image
    size=(224,224)
    image=ImageOps.fit(image,size,Image.ANTIALIAS)
    # convert this into numpy array
    image_array=np.asarray(image)
    # Normalise the Image - (0 to 255)
    normalise_image_array=(image_array.astype(np.float32)/127.0)-1
    # loading the image into the array
    data[0]=normalise_image_array
    # pass this data to model
    prediction=model.predict(data)
    print(prediction) # [[0.5,0.5,0.7,0.3]]
    # Decision Logic
    prediction=list(prediction[0])
    max_prediction=max(prediction)
    index_max=prediction.index(max_prediction)
    print(index_max)
    st.text("Expected Result: "+labels[index_max])

