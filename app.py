import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import streamlit as st
import numpy as py

model = load_model(r"C:\Users\annu\IdeaProjects\Image_classify.keras")
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

st.header("Image Classification Modal")
img_height = 180
img_width = 180 

image = st.text_input("Enter the fruit name", r'C:\Users\annu\IdeaProjects\beetroot.jpg')

image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr =  tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('Veg/Fruit in image is {} with accuracy of {:0.2f}'.format(data_cat[py.argmax(score)], py.max(score)*100))

