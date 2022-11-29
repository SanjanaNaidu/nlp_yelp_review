from django.urls import path
from . import views


urlpatterns = [
    path('dashboard', views.return_dashboard_stats, name='return_dashboard_stats'),
    path('search_keywords', views.return_search_keywords, name='return_search_keywords'),
]