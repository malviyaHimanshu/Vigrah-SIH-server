from email.policy import default
from pyexpat import model
from statistics import mode
from django.db import models
import uuid

# Create your models here.
class ImgData(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to='upload/ocr/', null=True, blank=True, default=None)
    img_pred = models.CharField(max_length=300, null=False)
    
    # def __str__(self):
    #     return self.id

