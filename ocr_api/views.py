from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import permissions
from rest_framework.parsers import MultiPartParser, FormParser

from main_api.service import MainApiService
from ocr_api.service import OCRApiService

from .serializers import ImgSerializer
from .models import ImgData

from keras.models import load_model
from keras.preprocessing import image

from django.conf import settings

# Line Segmentation

import cv2 as cv
from urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, Flatten
import os


# from models.get_preds import get_predictions

# Create your views here.
@api_view(['GET'])
def apiOverview(request):
  api_urls = {
      'Send Image to Model': '/ocr_api/send_image',
      'Access Image from database': '/ocr_api/get_image?<id>',
  }
  return Response(api_urls)


@api_view(['POST'])
def save_image(request):
    data = request.data
    ocr_service = OCRApiService()
    file = request.FILES
    res = ocr_service.send_image_to_model(data, file)
    return Response(res)


@api_view(['GET'])
def get_image(request):
    ocr_service = OCRApiService()
    img_id = request.query_params['id']
    img_obj = ocr_service.get_image_details(img_id)
    image_path = "./media/" + img_obj['image']
    img_obj['img_pred'] = get_preds(image_path)
    print("This is prediction --->> ", img_obj['img_pred'])
    print("This is result --->> ", img_obj)
    return Response(img_obj)





def findPeakRegions(hpp, divider=0.4):
    threshold = (np.max(hpp)-np.min(hpp))/divider
    peaks = []
    peaks_index = []
    for i, hppv in enumerate(hpp):
        if hppv > threshold:
            peaks.append([i, hppv])
    return peaks


def prepare(file):
    IMG_SIZE_X = 100
    IMG_SIZE_Y = 140
    img_array = cv.imread(file, cv.IMREAD_GRAYSCALE)
    new_array = cv.resize(img_array, (IMG_SIZE_X, IMG_SIZE_Y))
    return new_array.reshape(-1, 100, 140, 1)


def findPeakRegions(vpp, divider=0.88):
    threshold = (np.max(vpp)-np.min(vpp))/divider
    print(threshold)
    peaks = []
    peaks_index = []
    for i, vppv in enumerate(vpp):
        print(vppv)
        if vppv > threshold:
            peaks.append([i, vppv])
    return peaks


def get_preds(input_image):
    img_path = input_image

    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    hProj = np.sum(img,1)

    peaks = findPeakRegions(hProj, 0.44)
    print(peaks)
    peaksIndex = np.array(peaks)[:,0].astype(int)

    segmentedImg = np.copy(img)
    r,c = segmentedImg.shape

    for ri in range(r):
        if ri in peaksIndex:
            segmentedImg[ri, :] = 0

    hProjLines = np.sum(segmentedImg,1)

    hProjLines = np.append(hProjLines, [0, 0, 0])
    lines = []
    lIndexB = []
    lIndexE = []

    for ri in range(len(hProjLines)):
        if hProjLines[ri]!=0 and hProjLines[ri-1]==0 and hProjLines[ri-2]==0:
            lIndexB.append(ri)
        if hProjLines[ri]!=0 and hProjLines[ri+1]==0 and hProjLines[ri+2]==0:
            lIndexE.append(ri)
        
    for i in range(len(lIndexB)):
        print("pass")
        lines.append(img[lIndexB[i]:lIndexE[i],:])
        cv.imwrite("./models/Segmentation/lines/line{}.jpg".format(i+1),lines[i])

    class_names = ['a',
    'ba',
    'be',
    'bha',
    'bhe',
    'bhi',
    'bho',
    'bhu',
    'bhā',
    'bhī',
    'bhū',
    'bi',
    'bo',
    'bu',
    'bā',
    'ca',
    'ce',
    'cha',
    'che',
    'chi',
    'chu',
    'chā',
    'ci',
    'co',
    'cu',
    'cā',
    'cū',
    'da',
    'de',
    'dha',
    'dhe',
    'dhi',
    'dho',
    'dhu',
    'dhā',
    'dhī',
    'dhū',
    'di',
    'do',
    'du',
    'dā',
    'dī',
    'e',
    'ga',
    'ge',
    'gha',
    'ghe',
    'gho',
    'ghu',
    'ghā',
    'gi',
    'go',
    'gu',
    'gā',
    'ha',
    'he',
    'hi',
    'ho',
    'hu',
    'hā',
    'hī',
    'hū',
    'i',
    'ja',
    'je',
    'jha',
    'jhi',
    'jhā',
    'ji',
    'jo',
    'ju',
    'jā',
    'jī',
    'jū',
    'ka',
    'ke',
    'kha',
    'khe',
    'khi',
    'kho',
    'khu',
    'khā',
    'khī',
    'ki',
    'ko',
    'ku',
    'kā',
    'kī',
    'kū',
    'la',
    'le',
    'li',
    'lo',
    'lu',
    'lā',
    'lī',
    'lū',
    'ma',
    'me',
    'mi',
    'mo',
    'mu',
    'mā',
    'mī',
    'mū',
    'na',
    'ne',
    'ni',
    'no',
    'nu',
    'nā',
    'nī',
    'nū',
    'o',
    'pa',
    'pe',
    'pha',
    'phe',
    'phā',
    'pi',
    'po',
    'pu',
    'pā',
    'pī',
    'ra',
    're',
    'ri',
    'ro',
    'ru',
    'rā',
    'rī',
    'rū',
    'sa',
    'se',
    'si',
    'so',
    'su',
    'sā',
    'sī',
    'sū',
    'ta',
    'te',
    'tha',
    'the',
    'thi',
    'thu',
    'thā',
    'thī',
    'ti',
    'to',
    'tu',
    'tā',
    'tī',
    'tū',
    'u',
    'va',
    've',
    'vi',
    'vo',
    'vu',
    'vā',
    'vī',
    'vū',
    'ya',
    'ye',
    'yi',
    'yo',
    'yu',
    'yā',
    'yī',
    'yū',
    'ña',
    'ñe',
    'ño',
    'ñā',
    'ā',
    'śa',
    'śi',
    'śu',
    'śā',
    'ḍa',
    'ḍe',
    'ḍha',
    'ḍhe',
    'ḍhi',
    'ḍhā',
    'ḍhī',
    'ḍi',
    'ḍu',
    'ḍā',
    'ḍī',
    'ṇa',
    'ṇe',
    'ṇi',
    'ṇā',
    'ṇī',
    'ṣa',
    'ṣe',
    'ṣi',
    'ṣo',
    'ṣu',
    'ṣā',
    'ṭa','ṭe','ṭha','ṭhe','ṭhi','ṭhā','ṭhī','ṭhū','ṭi','ṭu','ṭā','ṭī']

    brahmi_dict = {'a':'𑀅','ā':'𑀆','i':'𑀇','ī':'𑀈','u':'𑀉','ū':'𑀊','e':'𑀏','ai':'𑀐','o':'𑀑','au':'𑀒','aṃ':'𑀅𑀁',
    'ka':'𑀓','kā':'𑀓𑀸','ki':'𑀓𑀺','kī':'𑀓𑀻','ku':'𑀓𑀼','kū':'𑀓𑀽','ke':'𑀓𑁂','kai':'𑀓𑁃','ko':'𑀓𑁄','kau':'𑀓𑁅','kaṃ':'𑀓𑀁',
    'kha':'𑀔​','khā':'𑀔𑀸','khi':'𑀔𑀺','khī':'𑀔𑀻','khu':'𑀔𑀼','khū':'𑀔𑀽','khe':'𑀔𑁂','khai':'𑀔𑁃','kho':'𑀔𑁄','khau':'𑀔𑁅','khaṃ':'𑀔𑀁',
    'ga':'𑀕​','gā':'𑀕𑀸','gi':'𑀕𑀺','gī':'𑀕𑀻','gu':'𑀕𑀼','gū':'𑀕𑀽','ge':'𑀕𑁂','gai':'𑀕𑁃','go':'𑀕𑁄','gau':'𑀕𑁅','gaṃ':'𑀕𑀁',
    'gha':'𑀖​','ghā':'𑀖𑀸','ghi':'𑀖𑀺','ghī':'𑀖𑀻','ghu':'𑀖𑀼','ghū':'𑀖𑀽','ghe':'𑀖𑁂','ghai':'𑀖𑁃','gho':'𑀖𑁄','ghau':'𑀖𑁅','ghaṃ':'𑀖𑀁',
    'ṅa':'𑀗​','ṅā':'𑀗𑀸','ṅi':'𑀗𑀺','ṅī':'𑀗𑀻','ṅu':'𑀗𑀼','ṅū':'𑀗𑀽','ṅe':'𑀗𑁂','ṅai':'𑀗𑁃','ṅo':'𑀗𑁄','ṅau':'𑀗𑁅','ṅaṃ':'𑀗𑀁',
    'ca':'𑀘​','cā':'𑀘𑀸','ci':'𑀘𑀺','cī':'𑀘𑀻','cu':'𑀘𑀼','cū':'𑀘𑀽','ce':'𑀘𑁂','cai':'𑀘𑁃','co':'𑀘𑁄','cau':'𑀘𑁅','caṃ':'𑀘𑀁',
    'cha':'𑀙​','chā':'𑀙𑀸','chi':'𑀙𑀺','chī':'𑀙𑀻','chu':'𑀙𑀼','chū':'𑀙𑀽','che':'𑀙𑁂','chai':'𑀙𑁃','cho':'𑀙𑁄','chau':'𑀙𑁅','chaṃ':'𑀙𑀁',
    'ja':'𑀚','jā':'𑀚𑀸','ji':'𑀚𑀺','jī':'𑀚𑀻','ju':'𑀚𑀼','jū':'𑀚𑀽','je':'𑀚𑁂','jai':'𑀚𑁃','jo':'𑀚𑁄','jau':'𑀚𑁅','jaṃ':'𑀚𑀁',
    'jha':'𑀛​','jhā':'𑀛𑀸','jhi':'𑀛𑀺','jhī':'𑀛𑀻','jhu':'𑀛𑀼','jhū':'𑀛𑀽','jhe':'𑀛𑁂','jhai':'𑀛𑁃','jho':'𑀛𑁄','jhau':'𑀛𑁅','jhaṃ':'𑀛𑀸𑀁',
    'ña':'𑀜​','ñā':'𑀜𑀸','ñi':'𑀜𑀺','ñī':'𑀜𑀻','ñu':'𑀜𑀼','ñū':'𑀜𑀽','ñe':'𑀜𑁂','ñai':'𑀜𑁃','ño':'𑀜𑁄','ñau':'𑀜𑁅','ñaṃ':'𑀜𑀁',
    'ṭa':'𑀝​','ṭā':'𑀝𑀸','ṭi':'𑀝𑀺','ṭī':'𑀝𑀻','ṭu':'𑀝𑀼','ṭū':'𑀝𑀽','ṭe':'𑀝𑁂','ṭai':'𑀝𑁃','ṭo':'𑀝𑁄','ṭau':'𑀝𑁅','ṭaṃ':'𑀝𑀁',
    'ṭha':'𑀞​','ṭhā':'𑀞𑀸','ṭhi':'𑀞𑀺','ṭhī':'𑀞𑀻','ṭhu':'𑀞𑀼','ṭhū':'𑀞𑀽','ṭhe':'𑀞𑁂','ṭhai':'𑀞𑁃','ṭho':'𑀞𑁄','ṭhau':'𑀞𑁅','ṭhaṃ':'𑀞𑀁',
    'ḍa':'𑀟​','ḍā':'𑀤𑀸','ḍi':'𑀟𑀺','ḍī':'𑀟𑀻','ḍu':'𑀟𑀼','ḍū':'𑀟𑀽','ḍe':'𑀟𑁂','ḍai':'𑀟𑁃','ḍo':'𑀟𑁄','ḍau':'𑀟𑁅','ḍaṃ':'𑀟𑀁',
    'ḍha':'𑀠​','ḍhā':'𑀠𑀸','ḍhi':'𑀠𑀺','ḍhī':'𑀠𑀻','ḍhu':'𑀠𑀼','ḍhū':'𑀠𑀽','ḍhe':'𑀠𑁂','ḍhai':'𑀠𑁃','ḍho':'𑀠𑁄','ḍhau':'𑀠𑁅','ḍhaṃ':'𑀠𑀁',
    'ṇa':'𑀡​','ṇā':'𑀡𑀸','ṇi':'𑀡𑀺','ṇī':'𑀡𑀻','ṇu':'𑀡𑀼','ṇū':'𑀡𑀽','ṇe':'𑀡𑁂','ṇai':'𑀡𑁃','ṇo':'𑀡𑁄','ṇau':'𑀡𑁅','ṇaṃ':'𑀡𑀁',
    'ta':'𑀢​','tā':'𑀢𑀸','ti':'𑀢𑀺','tī':'𑀢𑀻','tu':'𑀢𑀼','tū':'𑀢𑀽','te':'𑀢𑁂','tai':'𑀢𑁃','to':'𑀢𑁄','tau':'𑀢𑁅','taṃ':'𑀢𑀁',
    'tha':'𑀣​','thā':'𑀣𑀸','thi':'𑀣𑀺','thī':'𑀣𑀻','thu':'𑀣𑀼','thū':'𑀣𑀽','the':'𑀣𑁂','thai':'𑀣𑁃','tho':'𑀣𑁄','thau':'𑀣𑁅','thaṃ':'𑀣𑀁',
    'da':'𑀤​','dā':'𑀤𑀸','di':'𑀤𑀺','dī':'𑀤𑀻','du':'𑀤𑀼','dū':'𑀤𑀽','de':'𑀤𑁂','dai':'𑀤𑁃','do':'𑀤𑁄','dau':'𑀤𑁅','daṃ':'𑀤𑀁',
    'dha':'𑀥​','dhā':'𑀥𑀸','dhi':'𑀥𑀺','dhī':'𑀥𑀻','dhu':'𑀥𑀼','dhū':'𑀥𑀽','dhe':'𑀥𑁂','dhai':'𑀥𑁃','dho':'𑀥𑁄','dhau':'𑀥𑁅','dhaṃ':'𑀥𑀁',
    'na':'𑀦​','nā':'𑀦𑀸','ni':'𑀦𑀺','nī':'𑀦𑀻','nu':'𑀦𑀼','nū':'𑀦𑀽','ne':'𑀦𑁂','nai':'𑀦𑁃','no':'𑀦𑁄','nau':'𑀦𑁅','naṃ':'𑀦𑀁',
    'pa':'𑀧​','pā':'𑀧𑀸','pi':'𑀧𑀺','pī':'𑀧𑀻','pu':'𑀧𑀼','pū':'𑀧𑀽','pe':'𑀧𑁂','pai':'𑀧𑁃','po':'𑀧𑁄','pau':'𑀧𑁅','paṃ':'𑀧𑀁',
    'pha':'𑀨​','phā':'𑀨𑀸','phi':'𑀨𑀺','phī':'𑀨𑀻','phu':'𑀨𑀼','phū':'𑀨𑀽','phe':'𑀨𑁂','phai':'𑀨𑁃','pho':'𑀨𑁄','phau':'𑀨𑁅','phaṃ':'𑀨𑀁',
    'ba':'𑀩​','bā':'𑀩𑀸','bi':'𑀩𑀺','bī':'𑀩𑀻','bu':'𑀩𑀼','bū':'𑀩𑀽','be':'𑀩𑁂','bai':'𑀩𑁃','bo':'𑀩𑁄','bau':'𑀩𑁅','baṃ':'𑀩𑀁',
    'bha':'𑀪​','bhā':'𑀪𑀸','bhi':'𑀪𑀺','bhī':'𑀪𑀻','bhu':'𑀪𑀼','bhū':'𑀪𑀽','bhe':'𑀪𑁂','bhai':'𑀪𑁃','bho':'𑀪𑁄','bhau':'𑀪𑁅','bhaṃ':'𑀪𑀁',
    'ma':'𑀫​','mā':'𑀫𑀸','mi':'𑀫𑀺','mī':'𑀫𑀻','mu':'𑀫𑀼','mū':'𑀫𑀽','me':'𑀫𑁂','mai':'𑀫𑁃','mo':'𑀫𑁄','mau':'𑀫𑁅','maṃ':'𑀫𑀁',
    'ya':'𑀬​','yā':'𑀬𑀸','yi':'𑀬𑀺','yī':'𑀬𑀻','yu':'𑀬𑀼','yū':'𑀬𑀽','ye':'𑀬𑁂','yai':'𑀬𑁃','yo':'𑀬𑁄','yau':'𑀬𑁅','yaṃ':'𑀬𑀁',
    'ra':'𑀭​','rā':'𑀭​','ri':'𑀭𑀺','rī':'𑀭𑀻','ru':'𑀭𑀼','rū':'𑀭𑀽','re':'𑀭𑁂','rai':'𑀭𑁃','ro':'𑀭𑁄','rau':'𑀭𑁅','raṃ':'𑀭𑀁',
    'la':'𑀮​','lā':'𑀮𑀸','li':'𑀮𑀺','lī':'𑀮𑀻','lu':'𑀮𑀼','lū':'𑀮𑀽','le':'𑀮𑁂','lai':'𑀮𑁃','lo':'𑀮𑁄','lau':'𑀮𑁅','laṃ':'𑀮𑀁',
    'va':'𑀯​','vā':'𑀯𑀸','vi':'𑀯𑀺','vī':'𑀯𑀻','vu':'𑀯𑀼','vū':'𑀯𑀽','ve':'𑀯𑁂','vai':'𑀯𑁃','vo':'𑀯𑁄','vau':'𑀯𑁅','vaṃ':'𑀯𑀁',
    'śa':'𑀰','śā':'𑀰𑀸','śi':'𑀰𑀺','śī':'𑀰𑀻','śu':'𑀰𑀼','śū':'𑀰𑀽','śe':'𑀰𑁂','śai':'𑀰𑁃','śo':'𑀰𑁄','śau':'𑀰𑁅','śaṃ':'𑀰𑀁',
    'ṣa':'𑀱​','ṣā':'𑀱𑀸','ṣi':'𑀱𑀺','ṣī':'𑀱𑀻','ṣu':'𑀱𑀼','ṣū':'𑀱𑀽','ṣe':'𑀱𑁂','ṣai':'𑀱𑁃','ṣo':'𑀱𑁄','ṣau':'𑀱𑁅','ṣaṃ':'𑀱𑀁',
    'sa':'𑀲​','sā':'𑀲𑀸','si':'𑀲𑀺','sī':'𑀲𑀻','su':'𑀲𑀼','sū':'𑀲𑀽','se':'𑀲𑁂','sai':'𑀲𑁃','so':'𑀲𑁄','sau':'𑀲𑁅','saṃ':'𑀲𑀁',
    'ha':'𑀳​','hā':'𑀳𑀸','hi':'𑀳𑀺','hī':'𑀳𑀻','hu':'𑀳𑀼','hū':'𑀳𑀽','he':'𑀳𑁂','hai':'𑀳𑁃','ho':'𑀳𑁄','hau':'𑀳𑁅','haṃ':'𑀳𑀁'}


    char_model = keras.models.load_model('./models/best_val_loss_model.h5')

    line_dir = './models/Segmentation/lines/'

    final_pred = ""

    kth_img = 1

    line_breaks = []

    for filename in os.listdir(line_dir):
        img = cv.imread(line_dir + filename, cv.IMREAD_GRAYSCALE)

        vProj = np.sum(img,0)

        peaks = findPeakRegions(vProj,1.01)

        peaksIndex = np.array(peaks)[:,0].astype(int)

        segmentedImg = np.copy(img)
        r,c = segmentedImg.shape

        for ci in range(c):
            if ci in peaksIndex:
                segmentedImg[:,ci] = 0     

        vProjLines = np.sum(segmentedImg,0)
        vProjLines = np.append(vProjLines,[0,0,0])
        chars = []
        charsB = []
        charsE = []

        for ci in range(len(vProjLines)):
            if vProjLines[ci]!=0 and vProjLines[ci-1]==0:
                charsB.append(ci)
            if vProjLines[ci]!=0 and vProjLines[ci+1]==0:
                charsE.append(ci)

        for i in range(len(charsB)):
            chars.append(img[:,charsB[i]:charsE[i]]) 
            if len(chars[i][0]) != 0 : 
                cv.imwrite(f"./models/Segmentation/characters/char{kth_img}.png",chars[i])
    #             print(f"Segmentation/characters/char{kth_img}.png")
                kth_img += 1
        line_breaks.append(kth_img)


    char_dir = './models/Segmentation/characters/'

    num_chars = len([filename for filename in os.listdir(char_dir)])

    final_pred = ""

    for i in range(num_chars):
        if i in line_breaks:
            final_pred += '\n '
        img_path = f"{char_dir}char{i+1}.png"
        print(img_path)
        img = prepare(img_path)
        pred = char_model.predict([img])

        pred = list(pred[0])
        char_type = class_names[pred.index(max(pred))]
        final_pred += brahmi_dict[char_type]
        final_pred += " "

    print("This is the finalll predd\n", final_pred)
    return final_pred












