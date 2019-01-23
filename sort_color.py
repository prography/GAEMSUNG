import cv2
import os
from collections import Counter
from sklearn.cluster import KMeans

def dominant_color(image):
    hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
    hsv_img = cv2.resize(hsv_img, (25,25), interpolation = cv2.INTER_AREA)

    hsv_img = hsv_img.reshape((hsv_img.shape[0] * hsv_img.shape[1], 3))

    clt = KMeans(n_clusters = 4)
    labels = clt.fit_predict(hsv_img)

    label_counts = Counter(labels)

    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)


def filter_by_color(src_path, input_path, thresh=20):
    src = cv2.imread(src_path)
    
    src_color = dominant_color(src)

    colors = []
    for f in os.listdir(input_path):
        file_name = input_path + '/' + f
        comp = cv2.imread(file_name)

        comp_color = dominant_color(comp)

        
        if (abs(src_color[0]-comp_color[0]) <30) & (abs(src_color[1]-comp_color[1]) <30) & (abs(src_color[2]-comp_color[2]) <30):
            continue
        else:
            os.remove(file_name)

        
    

