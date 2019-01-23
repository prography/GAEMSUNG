import hashlib


from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.core import serializers
from django.http import request, HttpResponse, JsonResponse
from django.conf import settings
import json

import base64

from django.utils.crypto import random
from dl_module import *
from gaemsungapp.models import *
from django.views.decorators.csrf import csrf_exempt


#회원가입
@csrf_exempt
def join(request):
    request_data = ((request.body).decode('utf-8'))
    request_data = json.loads(request_data)

    user_name = request_data.get("user_id")
    password = request_data.get("pwd")

    age = request_data.get("age")
    gender= request_data.get("gender")

    new_user = User.objects.create_user(username=user_name,password = password)
    new_extra_info = User_extrainfo.objects.create(User = new_user, age = age,gender = gender)
    new_user.save()
    print(new_user.id)
    return JsonResponse({"success": "true"})

#로그인
@csrf_exempt
def login(request):
    request_data = ((request.body).decode('utf-8'))
    request_data = json.loads(request_data)

    username = request_data["user_id"]
    password = request_data["pwd"]
    user = authenticate(username=username, password=password)
    if user is not None:
        print(user.id)
        return JsonResponse({"success": True, "userid":user.id})
    else:
        return JsonResponse({"success": False, "userid": 0})

def convert_and_save(b64_string,key):
    if settings.SERVER == "LIVE":
        filename = "/home/cvserver/gaemsung/uploaded_image/"+ key + ".png"
    else :
        filename = "C:/Users/User/Pictures/tmp/"+key+".png"

    with open(filename, "wb") as fh:
        fh.write(base64.decodestring(bytes(b64_string, 'utf-8')))
        return filename

#이미지 검색하기
@csrf_exempt
def img_search(request):
    request_data = ((request.body).decode('utf-8'))
    request_data = json.loads(request_data)

    user_id = request_data["user_id"]
    img = request_data["image"]
    location = request_data['location']

    user = User.objects.get(id = user_id)

    if img is None:
        print("No valid request body, json missing!")
        return JsonResponse({'error': 'No valid request body, json missing!'})

    else:
        new_key =hashlib.sha1(str(random.random()).encode('utf-8')).hexdigest()[:15]
        decoded_img = convert_and_save(img,new_key)

    new_search_img = Search_Image.objects.create(User = user, img = decoded_img,location = location)

    search_loc = location+" 카페"
    search_img_url = new_search_img.img.url
    search_img_url = search_img_url.split("/")
    search_img = search_img_url[-2]+"/"+search_img_url[-1]
    find_img_list = find_similar_cafe(search_loc,search_img)
    for i in range(0,3):
        decoded_sub_img = find_img_list[i]['filename']
        sub_img_url = find_img_list[i]['url']
        new_sub_img = Sub_Image.objects.create(search_image=new_search_img,img = decoded_sub_img,url = sub_img_url)
        print(new_sub_img.img)
        print(new_sub_img.url)
    search_result = []
    for k in range(0,3):
        sub_img = Sub_Image.objects.filter(search_image = new_search_img).values('img')[k]
        print(sub_img)
        sub_img = sub_img["img"]
        sub_img = sub_img.split("/")
        sub_img = "http://203.246.84.126:8000/media/"+sub_img[-2]+"/"+sub_img[-1]
        sub_url = Sub_Image.objects.filter(search_image = new_search_img).values('url')[k]
        sub_url = sub_url["url"]
        search_result.append({"img" : sub_img,"url": sub_url})
    return JsonResponse({'subs':search_result})
# 딥러닝 method input : 검색어, 이미지 경로
# 딥러닝 method output : 리스트 형태의 (이미지 경로, url) 3개
#  여기서 이제 이미지를 넣으면 훈련 시키고, 결과값 subimg 생성, 저장하는거 값 되돌려주기


#mypage에서 username 얻어오기
@csrf_exempt
def get_mypage_name(request,user_id):
    user = User.objects.values('username').get(pk=user_id)
    data = user
    return JsonResponse(data, safe=False)

#mypage에서 user_info 얻어오기
@csrf_exempt
def get_mypage_info(request, user_id):
    user_info_list = []
    user = User.objects.values('username').get(pk=user_id)
    user_info = User_extrainfo.objects.values('age','gender').get(User__pk= user_id)
    return JsonResponse({'userInfo' : [{'username' : user.get('username') , 'age' : user_info.get('age') , 'gender' : user_info.get('gender')} ] })


@csrf_exempt
def get_mypage_img(requets, user_id):
    user_img = Search_Image.objects.filter(User__pk=user_id)
    lists = []
    for a in user_img:
        sub_lists = []
        sub_imgs = Sub_Image.objects.filter(search_image = a.id).values('img')
        sub_url = Sub_Image.objects.filter(search_image = a.id).values('url')
        for sub in sub_imgs:
             sub_img = sub.get('img')
             sub_img = sub_img.split("/")
             sub_img = "http://203.246.84.126:8000/media/"+sub_img[-2]+"/"+sub_img[-1]
             sub_url = sub.get('url')
             sub_lists.append({'img':sub_img , 'url':sub_url})
        main_img = str(a.img).split("/")
        main_img = "http://203.246.84.126:8000/media/uploaded_image/"+main_img[-1]
        lists.append({'main_img': main_img ,'date':  a.search_date,'subs': sub_lists })

    return JsonResponse({'mypage':lists})
