import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration
import os
from string import punctuation
from nltk.corpus import stopwords
import re
from image import search_images, download_images

'''
global uname, tokenizer, retriever, model
tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
print("done")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
print("done")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
print("done")
query = ["welcome to java world", "goiing to purchase laptop", "happy to see you", "welcome home"]
'''
#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))

#define function to clean text by removing stop words and other special symbols
def cleanText(data):
    data = data.split()
    data = [w for w in data if not w in stop_words]
    data = [word for word in data if len(word) > 3]
    data = ' '.join(data)
    return data

X = []
Y = []

if os.path.exists('ItineraryApp/static/features/X.npy'):
    X = np.load('ItineraryApp/static/features/X.npy')
    Y = np.load('ItineraryApp/static/features/Y.npy')

def scrapeImages(location):
    arr = location.split("_")
    name = arr[1]
    if os.path.exists('ItineraryApp/static/location_images/'+name) == False:
        os.makedirs('ItineraryApp/static/location_images/'+name)
        image_urls = search_images(name, num_images=5)
        download_images(image_urls, save_dir='ItineraryApp/static/location_images/'+name)    

search = []
rag = []
flag = False
for root, dirs, directory in os.walk('ItineraryApp/static/model'):
    for j in range(len(directory)):
        if directory[j] not in Y:
            with open(root+"/"+directory[j], "rb") as file:
                data = file.read()
            file.close()
            data = data.decode()
            data = data.strip('\n').strip().lower() 
            data = re.sub('[^a-z]+', ' ', data)
            data = cleanText(data)
            if len(data) > 2500:
                data = data[0:2500]
            inputs = tokenizer(data, return_tensors="pt")
            input_ids = inputs["input_ids"]
            question_hidden_states = model.question_encoder(input_ids)[0]
            question_hidden_states = question_hidden_states.detach().numpy().ravel()
            if len(X) > 0:
                X = X.tolist()
                Y = Y.tolist()
            Y.append(directory[j])
            X.append(question_hidden_states)
            X = np.asarray(X)
            Y = np.asarray(Y)
            scrapeImages(directory[j])
            flag = True
            print(directory[j])
if flag == True:
    np.save("ItineraryApp/static/features/X.npy", X)
    np.save("ItineraryApp/static/features/Y.npy", Y)
    
print(X)
print(Y)
print(X.shape)
print(Y.shape)

print(os.path.exists('ItineraryApp/static/model/india_china_5000.txt'))


        
