#!/usr/bin/env python
# coding: utf-8

# In[30]:


#3)Prepare and load your data
import json

#Load intents from JSON file
with open('oscar_intents.json') as f:
    data = json.load(f)

tags = []
patterns = []
responses = []
intents= []

#Loop through intents and extract patterns and responses
for intent in data['intents']:
    tags.append(intent['tag'])
    patterns.append(intent['patterns'])
    responses.append(intent['responses'])
    intents.append(intent)   

print('Tags:', tags)
print('\nPatterns:', patterns)
print('\nResponses:', responses)
print('\nIntents:', intents)


# In[31]:


#4)Pre-processing
from sklearn.preprocessing import LabelEncoder

lencd = LabelEncoder()

intent_labels = lencd.fit_transform(tags)

print(intent_labels)


# In[32]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=1100, oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)

sequences = tokenizer.texts_to_sequences(patterns)

max_length = 35
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

print(padded_sequences[0])


# In[33]:


#5)Deep learning training
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense

model_bot = Sequential()
model_bot.add(Embedding(input_dim=1100, output_dim=20, input_length=35))
model_bot.add(GlobalAveragePooling1D())
model_bot.add(Dense(16, activation='relu'))
model_bot.add(Dense(10, activation='sigmoid'))
model_bot.add(Dense(12, activation='softmax'))

model_bot.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_bot.summary()


# In[34]:


#Fit the model with 500 epochs
history_500 = model_bot.fit(padded_sequences, intent_labels, epochs=500, validation_split=0.2, batch_size=64)


# In[35]:


#Fit the model with 1000 epochs
history_1000 = model_bot.fit(padded_sequences, intent_labels, epochs=1000, validation_split=0.2, batch_size=64)


# In[36]:


#6)Testing your bot
import pickle

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('encoder.pickle', 'wb') as handle:
    pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

model_bot.save('model_intents.h5')


# In[ ]:




