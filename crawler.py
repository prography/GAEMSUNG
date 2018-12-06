from icrawler.builtin import GoogleImageCrawler

crawlwer = GoogleImageCrawler(storage={'root_dir': 'data_2/bicycle'})
crawlwer.crawl(keyword='자전거', max_num=200, file_idx_offset='auto')

crawlwer = GoogleImageCrawler(storage={'root_dir': 'data_2/bicycle'})
crawlwer.crawl(keyword='bicycle', max_num=200, file_idx_offset='auto')

crawlwer = GoogleImageCrawler(storage={'root_dir': 'data_2/bicycle'})
crawlwer.crawl(keyword='두발자전거', max_num=200, file_idx_offset='auto')

crawlwer = GoogleImageCrawler(storage={'root_dir': 'data_2/bicycle'})
crawlwer.crawl(keyword='보조바퀴 자전거', max_num=200, file_idx_offset='auto')

crawlwer = GoogleImageCrawler(storage={'root_dir': 'data_2/bicycle'})
crawlwer.crawl(keyword='예쁜 자전거', max_num=200, file_idx_offset='auto')

crawlwer = GoogleImageCrawler(storage={'root_dir': 'data_2/car'})
crawlwer.crawl(keyword='자동차', max_num=200, file_idx_offset='auto')

crawlwer = GoogleImageCrawler(storage={'root_dir': 'data_2/car'})
crawlwer.crawl(keyword='car', max_num=200, file_idx_offset='auto')

crawlwer = GoogleImageCrawler(storage={'root_dir': 'data_2/car'})
crawlwer.crawl(keyword='국산 자동차', max_num=200, file_idx_offset='auto')

crawlwer = GoogleImageCrawler(storage={'root_dir': 'data_2/car'})
crawlwer.crawl(keyword='수입 자동차', max_num=200, file_idx_offset='auto')

crawlwer = GoogleImageCrawler(storage={'root_dir': 'data_2/car'})
crawlwer.crawl(keyword='슈퍼카', max_num=200, file_idx_offset='auto')