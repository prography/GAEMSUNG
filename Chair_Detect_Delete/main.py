import os
import sys
from detect import detectfunc

import torchvision.transforms as T

def main():
	folder_path = 'datasets'
	
	for i in os.listdir(folder_path):
		print(i)
		for j in os.listdir(folder_path+'/'+i):
			print(j)
			file_path = folder_path+'/'+i+'/'+j
			#cmd = "python detect.py "+file_path
			#result = os.system(cmd)
			result = detectfunc(file_path)

			if (result==False) or (result==None):
				os.remove(file_path)
				print(j+"is removed")


	#result = detectfunc('testimg.png')
	#print(result)

if __name__ == "__main__":
	main()


