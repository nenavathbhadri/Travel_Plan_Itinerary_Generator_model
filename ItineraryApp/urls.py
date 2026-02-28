from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
               path("", views.index, name="index_root"),

               path("UserLogin.html", views.UserLogin, name="UserLogin"),	      
               path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
               path("RegisterAction", views.RegisterAction, name="RegisterAction"),
               path("Register.html", views.Register, name="Register"),
               path("TravelPlan.html", views.TravelPlan, name="TravelPlan"),
	       path("TravelPlanAction", views.TravelPlanAction, name="TravelPlanAction"),
	       path("TripHistory.html", views.TripHistory, name="TripHistory"),
	       # Admin routes
	       path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),
	       path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
	       path("AdminDashboard.html", views.AdminDashboard, name="AdminDashboard"),
	       path("AdminDeleteUser", views.AdminDeleteUser, name="AdminDeleteUser"),
	       path("AdminLogout", views.AdminLogout, name="AdminLogout"),
]
