# 🌍 TravelMind — AI-Powered Travel Plan Itinerary Generator

An intelligent web application that generates personalized, day-by-day travel itineraries using AI. Built with **Django**, **MySQL**, and powered by **Groq LLaMA 3.3 70B** and **Google Gemini** APIs, with a **RAG (Retrieval-Augmented Generation)** model for enhanced accuracy.

---

## ✨ Features

- 🤖 **AI-Generated Itineraries** — Get detailed day-by-day travel plans with activities, costs, food recommendations, and tips
- 🔄 **Dual AI Engine** — Groq LLaMA 3.3 70B (primary) + Google Gemini (fallback) for reliable generation
- 🧠 **RAG Model** — Retrieval-Augmented Generation using HuggingFace Transformers for grounded, high-quality plans
- 🗺️ **Destination Previews** — Browse popular destinations with preview cards that auto-fill trip details
- 📜 **Trip History** — Save and revisit all previously generated itineraries
- 👤 **User Authentication** — Secure registration & login with SHA-256 password hashing
- 🛡️ **Admin Panel** — Dashboard with user stats, trip analytics, top destinations, and user management
- 🌙 **Dark/Light Mode** — Sleek theme toggle with persistent preference
- 📱 **Fully Responsive** — Mobile-friendly design with glassmorphism effects

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.13, Django 6.0 |
| **Database** | MySQL + PyMySQL |
| **AI / ML** | Groq API (LLaMA 3.3 70B), Google Gemini API, HuggingFace Transformers, PyTorch |
| **NLP** | NLTK (stopword removal, text preprocessing) |
| **Frontend** | HTML5, CSS3 (Glassmorphism, CSS Grid/Flexbox), Vanilla JavaScript |
| **Typography** | Google Fonts (Inter, Playfair Display) |
| **Security** | SHA-256 hashing, Django CSRF, Session-based auth, python-dotenv |

---

## 📁 Project Structure

```
Travel Plan Itinerary Generator/
├── Itinerary/                    # Django project configuration
│   ├── __init__.py               # Registers PyMySQL as MySQLdb
│   ├── settings.py               # Django settings (DB, templates, static)
│   ├── urls.py                   # Root URL configuration
│   └── wsgi.py                   # WSGI entry point
├── ItineraryApp/                 # Main application
│   ├── views.py                  # All view functions (user, trip, admin)
│   ├── urls.py                   # App URL routing
│   └── templates/                # HTML templates
│       ├── base.html             # Base layout (navbar, theme, footer)
│       ├── index.html            # Landing page
│       ├── UserLogin.html        # User login form
│       ├── Register.html         # User registration form
│       ├── UserScreen.html       # User dashboard
│       ├── TravelPlan.html       # Trip planning page
│       ├── TripHistory.html      # Saved trips
│       ├── AdminLogin.html       # Admin login
│       └── AdminDashboard.html   # Admin dashboard
├── static/                       # Static assets
│   ├── style.css                 # Main CSS design system
│   ├── animations.css            # CSS animations
│   └── app.js                    # JavaScript (theme, validation, etc.)
├── .env                          # Environment variables (not in repo)
├── requirements.txt              # Python dependencies
├── database.txt                  # SQL schema reference
├── tech_stack.txt                # Detailed technology documentation
└── manage.py                     # Django management commands
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10+
- MySQL Server installed and running
- pip (Python package manager)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/nenavathbhadri/travel-plan-.git
cd travel-plan-
```

### Step 2 — Install Dependencies

```bash
pip install django pymysql python-dotenv groq google-generativeai torch transformers nltk
```

### Step 3 — Create MySQL Database

Open MySQL Workbench or CLI and run:

```sql
CREATE DATABASE itinerary;
```

> Tables (`register`, `trip_history`, `admin_users`) are auto-created when the server starts.

### Step 4 — Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=itinerary
```

### Step 5 — Run the Server

```bash
python manage.py runserver 8080
```

### Step 6 — Open in Browser

```
http://127.0.0.1:8080/
```

---

## 🗄️ Database Schema

### `register` — User Accounts

| Column | Type | Description |
|--------|------|-------------|
| username | VARCHAR(50) | Primary key |
| password | VARCHAR(128) | SHA-256 hashed |
| contact_no | VARCHAR(20) | Phone number |
| email | VARCHAR(50) | Email address |
| address | VARCHAR(65) | City / Address |

### `trip_history` — Saved Itineraries

| Column | Type | Description |
|--------|------|-------------|
| id | INT (AUTO) | Primary key |
| username | VARCHAR(50) | FK → register |
| source | VARCHAR(100) | Starting location |
| destination | VARCHAR(100) | Destination |
| budget | VARCHAR(20) | Budget amount |
| description | TEXT | Trip preferences |
| itinerary | LONGTEXT | AI-generated plan |
| created_at | TIMESTAMP | Generation time |

### `admin_users` — Admin Credentials

| Column | Type | Description |
|--------|------|-------------|
| id | INT (AUTO) | Primary key |
| username | VARCHAR(50) | Unique admin username |
| password | VARCHAR(128) | SHA-256 hashed |

---

## 🔐 Admin Panel

Access the admin panel at `/AdminLogin.html`

**Default Credentials:**
- Username: `admin`
- Password: `admin123`

**Admin Features:**
- 📊 Stats overview — total users, total trips, most popular destination
- 👥 User management — view and delete users
- 🗺️ Trip history — view all trips from all users
- 🏆 Top destinations — most visited destinations with trip counts

---

## 🔒 Security

- **SHA-256 Password Hashing** — Plain-text passwords are never stored
- **Django CSRF Protection** — All forms include CSRF tokens
- **Session-Based Auth** — Separate sessions for users and admins
- **Environment Variables** — API keys and DB credentials stored in `.env`

---

## 📸 Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Landing page with hero section |
| Login | `/UserLogin.html` | User login |
| Register | `/Register.html` | New user registration |
| Dashboard | `/UserScreen.html` | User dashboard |
| Plan Trip | `/TravelPlan.html` | AI trip planner with destination previews |
| Trip History | `/TripHistory.html` | View saved itineraries |
| Admin Login | `/AdminLogin.html` | Admin authentication |
| Admin Panel | `/AdminDashboard.html` | Admin dashboard |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is developed as a **Major Project** for academic purposes.

---

<p align="center">Made with ❤️ by <strong>Nenavath Bhadri</strong></p>
