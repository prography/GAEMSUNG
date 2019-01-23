from sort_color import dominant_color
from img_to_vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import os
import argparse
import shutil
import sys
import cv2

from datetime import datetime

import collections

import warnings
warnings.simplefilter("ignore",UserWarning)

import time
from crawler import crawl

from matplotlib.image import imread

def make_dir(input_path):
    try:
        os.makedirs(os.path.join(input_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory")
            raise

def remove_dir(input_path):
    try:
        shutil.rmtree(os.path.join(input_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to delete directory")
            raise

def convert_result_format(dict):
    format = {'filename':None, 'url':None}
    converted = [format.copy(), format.copy(), format.copy()]
    idx = 0

    for k, v in dict.items():
        filename = v[0]
        url = v[1]
        converted[idx]['filename'] = filename
        converted[idx]['url'] = url
        idx +=1

    return converted

def find_similar_cafe(keyword, input_path, save_path='crawlingData', max_file_num=20):

	##### 1. Crawling
	crawling_result = crawl(keyword, max_file_num)	

	##### 3. Similiarlity Matcing
	# 3.1 make input image to vector
	img2vec = Img2Vec()
	img = Image.open(input_path)
	#img_cv = cv2.imread(input_path)
	#img_color = dominant_color(img_cv)
	vec = img2vec.get_vec(img)

	sim_matric = {}
	index = 0

	colors = []
	elem_path_list = []
	
	''' TODO:: Corupted Image Check'''
	
	# 3.2 make similiarity matrix & dictionary
	path_idx = 0
	for dic in crawling_result:
		
		comp_img_path = dic['filename']
		comp_img_url = dic['url']

		check_img = imread(comp_img_path, 0)
		
		if check_img.shape[-1] == 3:
			comp_img = Image.open(comp_img_path)
			comp_img_vec = img2vec.get_vec(comp_img)
			
			sim = cosine_similarity(vec.reshape((1,-1)),comp_img_vec.reshape((1,-1)))[0][0]
				
			value_list = []
			value_list.append(comp_img_path)
			value_list.append(comp_img_url)

			sim_matric[sim] = value_list
		path_idx += 1
	
	ordered_dict = collections.OrderedDict(reversed(sorted(sim_matric.items())))
	resdic = dict(ordered_dict)
	
	# dict slicing
	final_dict = {}
	
	idx = 0
	for key in resdic.keys():

		if idx >2:
			break
				
		final_dict[key] = resdic[key]

		idx += 1

		
	converted = convert_result_format(final_dict)
	
	img_idx= 0	
	for dic in converted:
		save_img_path = dic['filename']
		uploaded_img_path = 'uploaded_image/' + input_path.split('.')[0] + '_' + str(img_idx) + '.jpg'
		dic['filename'] = uploaded_img_path
		cv_img = cv2.imread(save_img_path)
		cv2.imwrite(uploaded_img_path,cv_img)

		img_idx += 1

	print(converted)
