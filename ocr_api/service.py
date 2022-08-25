from .models import ImgData
from django.db import transaction

class OCRApiService:
  @classmethod
  @transaction.atomic
  def send_image_to_model(cls, img_data, file):
    if 'id' in img_data:
      img_object = ImgData.objects.get(id = img_data['id'])
    else:
      img_object = ImgData()

    if len(img_data.getlist('image[]')) > 0:
      for img in img_data.getlist('image[]'):
        img_object.image = img

    img_object.save()
    return img_object.id


  def get_image_details(cls, image_id):
    image_obj = ImgData.objects.filter(id = image_id).all().values(
      "id",
      "image",
      "img_pred"
    )[0]
    return image_obj