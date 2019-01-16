
from icrawler.builtin import GoogleImageCrawler
import logging

from detect import yoloDetector
from img_to_vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import os
import argparse
import shutil
import sys

from datetime import datetime

import collections

import warnings
warnings.simplefilter("ignore",UserWarning)

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

def find_similar_cafe(keyword, input_path, save_path='crawlingData', max_file_num=5):
    
    ##### 1. Crawling
    # 1.1 make folder to save crawling data
    if not (os.path.isdir(save_path)):
        make_dir(save_path)
    else:
        remove_dir(save_path)
        make_dir(save_path)
    
    # define log to file
    logging.basicConfig(filename='log.log', level=logging.INFO)
    logging.info('So should this')

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
    
    # 1.4 save image url
    url_list = [] 
    with open('crwling_log.log', 'r') as f1:
        log = f1.readlines()
        for lines in log:
            if 'downloader:image' in lines:
                image_url = lines.split('\t')[-1]
                url_list.append(image_url)
    f1.close()
    os.remove('crawling_log.log')

    ###### 2. Filtering.
    # 2.1 yolo class
    detector = yoloDetector()
    detector.load_weights()
    
    # 2.2 find image not including chair
    index = 0
    for f in os.listdir(save_path):
        elem_path = save_path+'/'+f
        res = detector.detect(elem_path)
            
        # 2.2.1 delete 
        if res is False:
            os.remove(elem_path)
            del url_list[index]

        index+=1

    ##### 3. Similiarlity Matcing
    # 3.1 make input image to vector
    img2vec = Img2Vec()
    img = Image.open(input_path)
    vec = img2vec.get_vec(img)
    
    sim_matric = {}
    index = 0

    # 3.2 make similiarity matrix & dictionary
    for f in os.listdir(save_path):
        elem_path = save_path + '/' + f
        # 3.2.1 make folder image to vector
        pic = Image.open(elem_path)
        pic_vec = img2vec.get_vec(pic)

        # 3.2.2 calculate cosine similarity
        sim = cosine_similarity(vec.reshape((1,-1)),pic_vec.reshape((1,-1)))[0][0]

        sim_matric[sim] = url_list[index]

        index += 1

    # 3.3. make result dictionary
    ordered_dict = collections.OrderedDict(reversed(sorted(sim_matric.items())))
    resdic = dict(ordered_dict)
    print(resdic)
    return resdic

