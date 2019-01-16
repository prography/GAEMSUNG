from icrawler.builtin import GoogleImageCrawler
import logging

# define log to file
logging.basicConfig(filename='info.log',level=logging.INFO)
logging.info('So should this')

# image crawling
google_crawler = GoogleImageCrawler(storage={'root_dir': 'save_image_dir'})
google_crawler.crawl(keyword='cat', max_num=5)

# open log file and delete other lines
with open('info.log', 'r') as f1:
    log = f1.readlines()

f2 = open('image_urls.txt', 'a')

for lines in log:
    if 'downloader:image' in lines:
        image_url = lines.split('\t')[-1]
        f2.write(image_url)

f1.close()
f2.close()
print("Done!!!")