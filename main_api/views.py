from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view

from main_api.service import MainApiService

from .serializers import InscriptionSerializer
from .models import InscriptionData
# Create your views here.

@api_view(['GET'])
def apiOverview(request):
    api_urls = {
        'List' : '/img-list/',
        'Detail View' : '/img-create/<int:id>',
        'Create' : '/img-create/',
        'Update' : '/img-update/<int:id>',
        'Delete' : '/img-delete/<int:id>',
    }

    return Response(api_urls)

@api_view(['GET'])
def ShowAll(request):
    imgs = InscriptionData.objects.all()
    serializer = InscriptionSerializer(imgs, many = True)
    return Response(serializer.data)


@api_view(['GET'])
def get_inscription_details(request):
    main_api_service = MainApiService()
    inscription_id = request.query_params['uid']
    inscription_list = main_api_service.get_inscriptions(inscription_id)
    return Response(inscription_list)