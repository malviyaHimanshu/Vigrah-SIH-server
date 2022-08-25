from readline import insert_text
from rest_framework import serializers

from .models import ImgData

class ImgSerializer(serializers.ModelSerializer):
    image = serializers.ImageField(allow_null=True)
    class Meta :
        model = ImgData
        fields = ['id', 'image']