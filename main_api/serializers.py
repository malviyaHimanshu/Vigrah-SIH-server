from readline import insert_text
from rest_framework import serializers

from .models import InscriptionData

class InscriptionSerializer(serializers.ModelSerializer):
    class Meta :
        model = InscriptionData
        fields   = '__all__'