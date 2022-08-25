from pyexpat import model
from rest_framework import viewsets, permissions

from . import serializers
from . import models

class OcrApiViewset(viewsets.ModelViewSet):
  queryset = models.ImgData.objects.all()
  serializer_class = serializers.ImgSerializer