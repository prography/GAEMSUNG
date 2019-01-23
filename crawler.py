from bs4 import BeautifulSoup
import requests
import urllib.request

def crawl(keyword, max_file_num):
	r = requests.get('https://search.naver.com/search.naver?where=image&sm=tab_jum&query=' + keyword)
	
	soup = BeautifulSoup(r.text, "html.parser")

	prev_title = None

	idx = 0


	result = []

	image_path = []
	image_url = []

	for link in soup.find_all('a'):
		
		if idx > max_file_num-1:
			break
		title = link.get('title')
		ref = link.get('href')

		is_blog = ref.split('//')

		if len(is_blog) > 1:
			key_blog = is_blog[1][0:4]

			if key_blog == 'blog':
				for src in soup.find_all('img'):
					img_alt = src.get('alt')
					img_src = src.get('data-source')

					if prev_title == img_alt:
						img_name = 'crawlingData/' + str(idx) + '.jpg'
						
						urllib.request.urlretrieve(img_src, img_name)
						
						return_dict = {}
						
						return_dict['filename'] = img_name
						return_dict['url'] = ref

						result.append(return_dict)
						idx += 1


		
		prev_title = title

	return result


