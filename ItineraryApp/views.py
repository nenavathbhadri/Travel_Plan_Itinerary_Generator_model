from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from dotenv import load_dotenv
load_dotenv()
import sqlite3
import json
from django.core.files.storage import FileSystemStorage
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
import requests as http_requests
import PIL.Image

global uname

#groq model to generate itinerary
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
print("Groq API configured (using llama3-70b)")

# Initialize SQLite database for user registration
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'db.sqlite3')
def init_user_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS register(
        username VARCHAR(50) PRIMARY KEY,
        password VARCHAR(50),
        contact_no VARCHAR(20),
        email VARCHAR(50),
        address VARCHAR(65)
    )''')
    conn.commit()
    conn.close()
init_user_db()

def call_gemini(prompt, max_retries=3):
    """Call Groq API to generate travel itinerary"""
    import time
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + GROQ_API_KEY
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4096
    }
    for attempt in range(max_retries + 1):
        resp = http_requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code == 429 and attempt < max_retries:
            wait_time = 15 * (attempt + 1)
            print("Rate limited. Waiting {} seconds before retry {}/{}...".format(wait_time, attempt + 1, max_retries))
            time.sleep(wait_time)
            continue
        if resp.status_code != 200:
            print("Groq API error {}: {}".format(resp.status_code, resp.text))
        resp.raise_for_status()
        break
    data = resp.json()
    return data["choices"][0]["message"]["content"]

stop_words = set(stopwords.words('english'))

#define function to clean text by removing stop words and other special symbols
def cleanText(data):
    data = data.split()
    data = [w for w in data if not w in stop_words]
    data = [word for word in data if len(word) > 3]
    data = ' '.join(data)
    return data
    
#function to scrape images from net for selected itinerary
def scrapeImages(location):
    arr = location.split("_")
    if len(arr) > 1:
        name = arr[1]
    else:
        name = location

    if os.path.exists('ItineraryApp/static/location_images/'+name) == False:
        os.makedirs('ItineraryApp/static/location_images/'+name)
        image_urls = search_images(name, num_images=5)
        download_images(image_urls, save_dir='ItineraryApp/static/location_images/'+name)

# Lazy loader for models

class ModelLoader:
    _instance = None
    
    def __init__(self):
        self.tokenizer = None
        self.retriever = None
        self.model = None
        self.X = []
        self.Y = []
        self.models_loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelLoader()
        return cls._instance

    def load_models(self):
        if self.models_loaded:
            return

        print("Loading models...")
        global stop_words
        stop_words = set(stopwords.words('english'))
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
        
        print("Loading retriever...")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
        
        print("Loading RAG model...")
        self.model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=self.retriever)
        
        self.load_data()
        self.models_loaded = True
        print("Models loaded successfully.")

    def load_data(self):
        print("Loading data features...")
        if os.path.exists('ItineraryApp/static/features/X.npy'):
            self.X = np.load('ItineraryApp/static/features/X.npy')
            self.Y = np.load('ItineraryApp/static/features/Y.npy')
            # Convert to list if they are numpy arrays to allow appending
            if isinstance(self.X, np.ndarray):
                 self.X = self.X.tolist()
            if isinstance(self.Y, np.ndarray):
                 self.Y = self.Y.tolist()
        else:
            self.X = []
            self.Y = []

        flag = False
        # Ensure directory exists
        model_dir = 'ItineraryApp/static/model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for root, dirs, directory in os.walk(model_dir):
            for j in range(len(directory)):
                if directory[j].lower() not in self.Y:
                    try:
                        with open(os.path.join(root, directory[j]), "rb") as file:
                            data = file.read()
                        data = data.decode('utf-8', errors='ignore') # Added error handling
                        data = data.strip('\n').strip().lower() 
                        data = re.sub('[^a-z]+', ' ', data)
                        data = cleanText(data)
                        if len(data) > 2500:
                            data = data[0:2500]
                        
                        inputs = self.tokenizer(data, return_tensors="pt")
                        input_ids = inputs["input_ids"]
                        question_hidden_states = self.model.question_encoder(input_ids)[0]
                        question_hidden_states = question_hidden_states.detach().numpy().ravel()
                        
                        self.Y.append(directory[j].lower())
                        self.X.append(question_hidden_states)
                        
                        # scrapeImages might need to be robust
                        try:
                            scrapeImages(directory[j])
                        except Exception as e:
                            print(f"Error scraping images for {directory[j]}: {e}")

                        flag = True
                        print(f"Processed {directory[j]}")
                    except Exception as e:
                        print(f"Error processing file {directory[j]}: {e}")

        if flag:
            # Convert back to numpy for saving/processing if needed, or keep as list until needed
            # The original code treated X as list for appending then converted to numpy
            # We'll save as numpy
            np.save("ItineraryApp/static/features/X.npy", np.array(self.X))
            np.save("ItineraryApp/static/features/Y.npy", np.array(self.Y))
        
        # Ensure X and Y are lists for further processing if needed, or handle as numpy
        # The original code seems to switch back and forth. Let's keep them as numpy arrays for the dot product
        self.X = np.array(self.X) if isinstance(self.X, list) else self.X
        self.Y = np.array(self.Y) if isinstance(self.Y, list) else self.Y
        
        # Start background thread or ensure this doesn't block if possible? 
        # For now, just running it here as it was original logic, just deferred.
        
        print(f"Data loaded. X shape: {self.X.shape if hasattr(self.X, 'shape') else len(self.X)}")


# Helper to ensure models are loaded
def get_model_loader():
    loader = ModelLoader.get_instance()
    if not loader.models_loaded:
        loader.load_models()
    return loader


def TravelPlanAction(request):
    if request.method == 'POST':
        # Ensure models are loaded
        loader = get_model_loader()
        global uname

        source = request.POST.get('t1', False)
        destination = request.POST.get('t2', False)
        budget = request.POST.get('t3', False)
        desc = request.POST.get('t4', False)
        name = source+"_"+destination+"_"+budget+".txt"
        print(loader.Y)

        print(name)
        tt = name.lower() not in loader.Y
        print(tt)
        if name.lower() not in loader.Y:

            prompt = f"""Generate a trip plan for from {source} to {destination} with a budget of ${budget}.
             We are interested in a mix of historical sightseeing, cultural experiences, travel websites, hotel booking platforms, tourist guides,
             transportation schedules, weather forecasts and delicious food.
             Provide a detailed itinerary for hotels and flights
            """
            plan = call_gemini(prompt)
            with open("ItineraryApp/static/model/"+source+"_"+destination+"_"+budget+".txt", "wb") as file:
                file.write(plan.encode())
            file.close()
            # Reload data to include new plan
            loader.load_data()

        data = source+" "+destination+" "+desc
        data = data.strip('\n').strip().lower()
        data = re.sub('[^a-z]+', ' ', data)
        data = cleanText(data)
        if len(data) > 2500:
            data = data[0:2500]
        inputs = loader.tokenizer(data, return_tensors="pt")

        input_ids = inputs["input_ids"]
        query = loader.model.question_encoder(input_ids)[0]

        query = query.detach().numpy().ravel()
        plan_name = ""
        max_score = 0
        X = loader.X
        Y = loader.Y
        for i in range(len(X)):
            predict_score = dot(X[i], query)/(norm(X[i])*norm(query))
            if predict_score > max_score and destination.lower() in Y[i]:
                max_score = predict_score
                plan_name = Y[i]

        data = ""        
        if not plan_name:
            # Fallback or error handling if no plan found
            data = "No matching itinerary found."
        else:
            with open('ItineraryApp/static/model/'+plan_name, "r") as file:
                for line in file:
                    values = line.strip()
                    if len(values) == 0:
                        data += "<br/>"
                    else:
                        data += values+"<br/>"


        output = '<br/><table border=0 align=center><tr>'
        for root, dirs, directory in os.walk('ItineraryApp/static/location_images/'+destination.lower()):
            for j in range(len(directory)):
                output += '<td><img src="static/location_images/'+destination.lower()+'/'+directory[j]+'" width="200" height="200"/></td>'
        output += "</tr></table><br/><br/><br/><br/>"      
        data += output
        context= {'data': data}
        return render(request, 'UserScreen.html', context)
        
def TravelPlan(request):
    if request.method == 'GET':
       return render(request, 'TravelPlan.html', {})

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})   

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT username, password FROM register WHERE username=? AND password=?", (username, password))
        rows = cur.fetchall()
        for row in rows:
            uname = username
            index = 1
            break
        con.close()
        if index == 1:
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'UserLogin.html', context)        
    
def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)        
        
        status = "none"
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT username FROM register")
        rows = cur.fetchall()
        for row in rows:
            if row[0] == username:
                status = "Username already exists"
                break
        if status == "none":
            cur.execute("INSERT INTO register VALUES(?,?,?,?,?)", (username, password, contact, email, address))
            con.commit()
            print(cur.rowcount, "Record Inserted")
            if cur.rowcount == 1:
                status = "Signup task completed"
        con.close()
        context= {'data': status}
        return render(request, 'Register.html', context)

