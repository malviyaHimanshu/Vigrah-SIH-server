from pyexpat import model
from statistics import mode
from django.db import models
import uuid

# Create your models here.
class InscriptionData(models.Model):
    uid = models.CharField(max_length=100, null=False, blank=False, default='')
    name = models.CharField(max_length=300, null=False, default='')
    inscription_image = models.ImageField(null = True, blank = True, upload_to = 'images/')
    location_lat = models.FloatField()
    location_long = models.FloatField()
    location = models.CharField(max_length=300, null=True, blank=True)
    person = models.CharField(max_length=100, null=True, blank=True)
    langauge = models.CharField(max_length=100, null=True, blank=True)
    script = models.CharField(max_length=100, null=True, blank=True)
    time_period = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return self.name

