from django.urls import path
from . import views

urlpatterns = [
    path('', views.apiOverview, name = 'apiOverview'),
    path('get_inscriptions', views.ShowAll, name='get_inscriptions'),
    path('get_inscription_details', views.get_inscription_details, name='get_inscription_details'),
]
