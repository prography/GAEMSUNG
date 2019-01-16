
from icrawler.builtin import GoogleImageCrawler
from detect import yoloDetector
from img_to_vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import os
import argparse
import shutil
import sys

from datetime import datetime

def make_dir(input_path):
	try:
		if not(os.path.isdir(input_path)):
			os.makedirs(os.path.join(input_path))
	except OSError as e:
		if e.errno != errno.EEXIST:
			print("Failed to create directory")
			raise

def find_similar_cafe(keyword, input_path, save_path='crawlingData', max_file_num=5):
    
    ### 1. Crawling
    # 1.1 make folder to save crawling data
    make_dir(save_path)
    
    '''
    # 1.2 make crawler class
    google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=2,
    downloader_threads=4,
    storage={'root_dir': save_path})

    # 1.3 set crawling filter
    filters = dict(
        license='commercial,modify',
    )
    
    google_crawler.crawl(keyword=keyword, filters=filters, max_num=max_file_num, file_idx_offset='auto')
    '''
    ### 2. Filtering.
    # 2.1 yolo class
    detector = yoloDetector()
    detector.load_weights()
    
    # 2.2 find image not including chair
    for f in os.listdir(save_path):
        elem_path = save_path+'/'+f
        res = detector.detect(elem_path)
        
        # 2.2.1 delete 
        if res is False:
            os.remove(elem_path)
            print('filtered : ', elem_path)

    ### 3. Similiarlity Matcing
    img2vec = Img2Vec()
    img = Image.open(input_path)
    vec = img2vec.get_vec(img)
    
    sim_matric = []
    for f in os.listdir(save_path):
        elem_path = save_path + '/' + f
        pic = Image.open(elem_path)
        pic_vec = img2vec.get_vec(pic)

        sim = cosine_similarity(vec.reshape((1,-1)),pic_vec.reshape((1,-1)))[0][0]
        print(sim)
        sim_matric.append(sim)

    sim_matric.sort(reverse=True)

    print(sim_matric[:3])
    return sim_matric[:3]
        
        










