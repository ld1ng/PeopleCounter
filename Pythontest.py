import os
import cv2
import time
import datetime
import urllib.request
from PIL import Image
from imageai.Detection import ObjectDetection
import requests
import numpy as np
# 实例化人脸分类器
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 准备摄像机上的视频捕获
cap = cv2.VideoCapture(0)
cv2.namedWindow("PyCam")


def main():
	# 当前工作目录
	execution_path = os.getcwd()

	# 初始化ObjectDetection()类
	detector = ObjectDetection()
	detector.setModelTypeAsRetinaNet()
	detector.setModelPath("resnet50_coco_best_v2.0.1.h5")
	detector.loadModel(detection_speed="fastest") 
	#options=["normal", "fast", "faster", "fastest", "flash"]

	# 选择要检测的对象类型
	custom_objects = detector.CustomObjects(person=True)

	# 初始化计数器
	img_counter = 0

	file_to_delete = ""

	start_time = 0

	while(True):		
		# 捕获当前帧
		ret, frame = cap.read()
		img = cv2.flip(frame, 2)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		#找到人脸，并将人脸的坐标、矩形大小用vector保存
		#scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%
		faces = faceCascade.detectMultiScale(
			gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)#表示人脸识别的最小矩形大小
		)

		#「标记」图像中的人脸
		for (x,y,w,h) in faces:
		#255，0，0是蓝色，蓝绿红；2是划线粗细
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]

		#统计正脸的个数
		if len(faces):
			careful=format(len(faces))
		else:
			careful=0
		# 显示结果帧
		cv2.imshow('frame',img)

		# 按退出键退出程序
		k = cv2.waitKey(1)
		if k % 256 == 27:
			break

		if time.time() - start_time > 30:
			# 记录此特定帧捕获的时间.
			current_time = str(datetime.datetime.now())

			# 定义当前的输入文件和输出文件以运行检测器脚本
			inputfile = "images/in/input_frame_{}.png".format(img_counter)
			outputfile = "images/out/output_frame_{}.png".format(img_counter)

			# 用于命名图像文件的增量img_counter
			img_counter += 1

			# 从当前帧创建新图像文件
			cv2.imwrite(inputfile, frame)

			# 检测对象并创建一个输出文件，其中包含对象周围的正方形
			detections = detector.detectCustomObjectsFromImage(
				custom_objects=custom_objects,
				input_image=inputfile,
				output_image_path=outputfile,
				minimum_percentage_probability=33
			)

			# 以字符串形式获取n阵列的检测长度
			num_detections = str(len(detections)) 

			#统计低头或侧脸个数
			n=int(num_detections)-int(careful)

			#计算专注度
			def quality():
				if(n>=0):
					if (int(careful)/int(num_detections)>=0.85):
						print ("课堂质量:优")
					if(0.5<=int(careful)/int(num_detections)<0.85):
						print ("课堂质量:中")
					if(int(careful)/int(num_detections)<0.5):
						print ("课堂质量:差")
				if(n<0):
					print ("检测错误")
				

			# 输出
			print("[%s] 检测人数: %s , 其中 %d 人上课不专心" %(current_time, num_detections, n))
			quality()

			# 删除输入文件
			os.remove(inputfile)

			# 删除之前的输出文件
			try:
				os.remove(file_to_delete)
			except:
				# 什么都不做
				pass

			# 将要删除的下一个文件设置为当前输出文件
			file_to_delete = outputfile

			# 为了演示效果,打开用户设备上的输出图像
			Image.open(outputfile).show()
			
			start_time = time.time()

		# 让CPU休眠0.2秒以保持能量
		time.sleep(0.2)

if __name__ == "__main__":
	main()
