from icrawler.builtin import GoogleImageCrawler
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

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('key', type=str, default=None, help='search word')
    parser.add_argument('--save_path', type=str, default='crawlingData', help='root path to crawling directory')
    parser.add_argument('--image_size', type=str, default=None, help='image size filter')
    parser.add_argument('--color', type=str, default=None, help='image color filter')
    parser.add_argument('--period', type=str, default=None, help='2017_11_27-2018_11_27')
    
    return parser.parse_args(argv)

def main(args):

    # default : 1 year
    if args.period is None:
        i_p_year = datetime.today().year - 1
        i_p_month = datetime.today().month
        i_p_day = datetime.today().day

        f_p_year = datetime.today().year
        f_p_month = datetime.today().month
        f_p_day = datetime.today().day

    else:
        # cutting args.period value
        i_period = args.period.split('-')[0]
        f_period = args.period.split('-')[1]

        i_p_year = i_period.split('_')[0]
        i_p_month = i_period.split('_')[1]
        i_p_day = i_period.split('_')[2]

        f_p_year = f_period.split('_')[0]
        f_p_month = f_period.split('_')[1]
        f_p_day = f_period.split('_')[2]


    google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=2,
    downloader_threads=4,
    storage={'root_dir': args.save_path})

    filters = dict(
        #size = args.image_size,
        #color= args.color,
        license='commercial,modify',
        date=((i_p_year, i_p_month, i_p_day), (f_p_year, f_p_month, f_p_day)))

    google_crawler.crawl(keyword=args.key, filters=filters, max_num=100, file_idx_offset=0)    


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


    

