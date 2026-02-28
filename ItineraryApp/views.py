from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from dotenv import load_dotenv
load_dotenv()
import pymysql
import hashlib
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

# username stored in session (see UserLoginAction)

#groq model to generate itinerary
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
print("Groq API configured (using llama3-70b)")

# ============================================
# MySQL Database Configuration
# ============================================
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "1234")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "itinerary")

def get_db_connection():
    """Get a MySQL database connection"""
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor
    )

def hash_password(password):
    """Hash password using SHA-256 for secure storage"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def init_mysql_db():
    """Initialize MySQL database tables"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS register(
            username VARCHAR(50) PRIMARY KEY,
            password VARCHAR(128),
            contact_no VARCHAR(20),
            email VARCHAR(50),
            address VARCHAR(65)
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS trip_history(
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50),
            source VARCHAR(100),
            destination VARCHAR(100),
            budget VARCHAR(20),
            description TEXT,
            itinerary LONGTEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(username) REFERENCES register(username)
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS admin_users(
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE,
            password VARCHAR(128)
        )''')
        conn.commit()
        conn.close()
        print("MySQL database initialized successfully.")
    except Exception as e:
        print(f"MySQL init error: {e}")

init_mysql_db()

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
            data = "<p>No matching itinerary found. Please try again with different details.</p>"
        else:
            with open('ItineraryApp/static/model/'+plan_name, "r") as file:
                for line in file:
                    values = line.strip()
                    if len(values) == 0:
                        data += "<br/>"
                    else:
                        # Format headings with proper HTML tags
                        if values.startswith('**') and values.endswith('**'):
                            data += '<h3 style="margin-top:1rem;">' + values.strip('*') + '</h3>'
                        elif values.startswith('# '):
                            data += '<h3 style="margin-top:1rem;">' + values[2:] + '</h3>'
                        elif values.startswith('## '):
                            data += '<h4 style="margin-top:0.75rem;">' + values[3:] + '</h4>'
                        else:
                            data += '<p style="margin-bottom:0.25rem;">' + values + '</p>'

        # Build images list for the gallery grid
        images = []
        img_dir = 'ItineraryApp/static/location_images/' + destination.lower()
        if os.path.exists(img_dir):
            for root, dirs, directory in os.walk(img_dir):
                for j in range(len(directory)):
                    images.append('location_images/' + destination.lower() + '/' + directory[j])

        # Save trip to MySQL history
        username = request.session.get('username', '')
        if username:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO trip_history (username, source, destination, budget, description, itinerary) VALUES (%s, %s, %s, %s, %s, %s)",
                    (username, source, destination, budget, desc, data)
                )
                conn.commit()
                conn.close()
                print(f"Trip saved to history for user: {username}")
            except Exception as e:
                print(f"Error saving trip history: {e}")

        context = {'data': data, 'images': images, 'username': username}
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
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        hashed_password = hash_password(password)
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT username, password FROM register WHERE username=%s", (username,))
            row = cur.fetchone()
            conn.close()
            if row and row['password'] == hashed_password:
                request.session['username'] = username
                # Load trip history count for welcome message
                context = {'data': 'welcome ' + username, 'username': username}
                return render(request, 'UserScreen.html', context)
            else:
                context = {'data': 'Invalid username or password. Please try again.'}
                return render(request, 'UserLogin.html', context)
        except Exception as e:
            print(f"Login error: {e}")
            context = {'data': 'Database error. Please try again later.'}
            return render(request, 'UserLogin.html', context)
    
def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        hashed_password = hash_password(password)
        status = "none"
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT username FROM register WHERE username=%s", (username,))
            row = cur.fetchone()
            if row:
                status = "Username already exists"
            else:
                cur.execute(
                    "INSERT INTO register (username, password, contact_no, email, address) VALUES (%s, %s, %s, %s, %s)",
                    (username, hashed_password, contact, email, address)
                )
                conn.commit()
                print(cur.rowcount, "Record Inserted into MySQL")
                if cur.rowcount == 1:
                    status = "Account created successfully! You can now sign in."
            conn.close()
        except Exception as e:
            print(f"Registration error: {e}")
            status = "Registration failed. Please try again."
        context = {'data': status}
        return render(request, 'Register.html', context)

def TripHistory(request):
    """View trip history for the logged-in user"""
    username = request.session.get('username', '')
    if not username:
        return render(request, 'UserLogin.html', {'data': 'Please log in to view your trip history.'})
    
    trips = []
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, source, destination, budget, description, itinerary, created_at FROM trip_history WHERE username=%s ORDER BY created_at DESC",
            (username,)
        )
        trips = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"Trip history error: {e}")
    
    context = {'trips': trips, 'username': username}
    return render(request, 'TripHistory.html', context)

# ============================================
# ADMIN PANEL VIEWS
# ============================================
def AdminLogin(request):
    if request.method == 'GET':
        return render(request, 'AdminLogin.html', {})

def AdminLoginAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', '')
        password = request.POST.get('t2', '')
        hashed_password = hash_password(password)
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT username, password FROM admin_users WHERE username=%s", (username,))
            row = cur.fetchone()
            conn.close()
            if row and row['password'] == hashed_password:
                request.session['admin_username'] = username
                from django.shortcuts import redirect
                return redirect('AdminDashboard')
            else:
                context = {'data': 'Invalid admin credentials.'}
                return render(request, 'AdminLogin.html', context)
        except Exception as e:
            print(f"Admin login error: {e}")
            context = {'data': 'Database error. Please try again.'}
            return render(request, 'AdminLogin.html', context)

def AdminDashboard(request):
    admin = request.session.get('admin_username', '')
    if not admin:
        return render(request, 'AdminLogin.html', {'data': 'Please log in as admin.'})
    
    users = []
    trips = []
    stats = {}
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Get all users
        cur.execute("SELECT username, contact_no, email, address FROM register ORDER BY username")
        users = cur.fetchall()
        # Get all trips
        cur.execute("SELECT t.id, t.username, t.source, t.destination, t.budget, t.created_at FROM trip_history t ORDER BY t.created_at DESC")
        trips = cur.fetchall()
        # Stats
        cur.execute("SELECT COUNT(*) as cnt FROM register")
        stats['total_users'] = cur.fetchone()['cnt']
        cur.execute("SELECT COUNT(*) as cnt FROM trip_history")
        stats['total_trips'] = cur.fetchone()['cnt']
        cur.execute("SELECT destination, COUNT(*) as cnt FROM trip_history GROUP BY destination ORDER BY cnt DESC LIMIT 5")
        stats['top_destinations'] = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"Admin dashboard error: {e}")

    top_dest = ''
    if stats.get('top_destinations'):
        top_dest = stats['top_destinations'][0]['destination']
        for d in stats['top_destinations']:
            d['trip_count'] = d['cnt']

    context = {'users': users, 'trips': trips, 'stats': stats, 'admin': admin, 'top_dest': top_dest}
    return render(request, 'AdminDashboard.html', context)

def AdminDeleteUser(request):
    admin = request.session.get('admin_username', '')
    if not admin:
        return render(request, 'AdminLogin.html', {'data': 'Please log in as admin.'})
    
    if request.method == 'POST':
        username = request.POST.get('username', '')
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            # Delete trip history first (FK constraint)
            cur.execute("DELETE FROM trip_history WHERE username=%s", (username,))
            cur.execute("DELETE FROM register WHERE username=%s", (username,))
            conn.commit()
            conn.close()
            print(f"Admin deleted user: {username}")
        except Exception as e:
            print(f"Admin delete error: {e}")
    
    return AdminDashboard(request)

def AdminLogout(request):
    if 'admin_username' in request.session:
        del request.session['admin_username']
    return render(request, 'AdminLogin.html', {})
