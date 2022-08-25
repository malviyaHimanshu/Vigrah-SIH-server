from .models import InscriptionData

class MainApiService:
  @classmethod
  def get_inscriptions(cls, IID):
    inscription_list = InscriptionData.objects.filter(uid = IID).values(
      "uid",
      "name",
      "inscription_image",
      "location_lat",
      "location_long",
      "location",
      "person",
      "langauge",
      "script",
      "time_period"
    )
    return list(inscription_list)
    # for i in list(inscription_list):

