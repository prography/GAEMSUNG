from icrawler.builtin import GoogleImageCrawler

crawlwer = GoogleImageCrawler(storage={'root_dir': 'data/bicycle'})
crawlwer.crawl(keyword='bicycle', max_num=1000)