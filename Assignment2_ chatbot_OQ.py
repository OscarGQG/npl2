#!/usr/bin/env python
# coding: utf-8

# In[24]:


#6)Testing your bot
import tensorflow as tf
import pickle
import json
import random

#Load tokenizer and encoder
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('encoder.pickle', 'rb') as handle:
    encoder = pickle.load(handle)

#Load model
model = tf.keras.models.load_model('model_intents.h5')

#Load intents file
with open('oscar_intents.json') as file:
    intents = json.load(file)


# In[25]:


#Function to predict intent
def predict_intent(text):    
    sequence = tokenizer.texts_to_sequences([text])    
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=35, padding="post")    
    predictions = model.predict(sequence)    
    index = tf.argmax(predictions, axis=1).numpy()[0]    
    intent = encoder.inverse_transform([index])[0]
    return intent

#Function to get response
def get_response(intent):    
    responses = [i['responses'] for i in intents['intents'] if i['tag']==intent][0]    
    response = random.choice(responses)
    return response


# In[29]:


#Run Chatbot
while True:    
    user_input = input("Customer: ")    
    intent = predict_intent(user_input)    
    response = get_response(intent)
    
    #Check for exit command
    if user_input.lower() == "exit":
        break
    else: 
        print("ChatBot: " + response)


# In[ ]:




