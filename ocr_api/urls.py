from django.urls import path, include
from . import api
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register("OcrApi", api.OcrApiViewset)

urlpatterns = [
    path('', views.apiOverview, name='apiOverview'),
    path('send_image', views.save_image, name='send_image'),
    path('get_image', views.get_image, name='get_image'),
]
