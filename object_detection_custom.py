import tensorflow as tf
import cv2
import numpy as np
import sys
import argparse

def generate_graph():
	#loading saved inference graph
	with tf.gfile.GFile('trained_graphs/frozen_inference_graph.pb','rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	return graph_def
	
def detection(graph_def, input_path):
	#running tensorflow session
	with tf.Session() as sess:
		sess.graph.as_default()
		#importing saved inference graph to current session graph
		tf.import_graph_def(graph_def,name='')

		#reading image
		image = cv2.imread(input_path)
		
		inp = cv2.resize(image,(300,300))
		inp = inp[:,:,[2,1,0]] #converting BGR format to RGB format

		#running model
		output = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
							sess.graph.get_tensor_by_name('detection_scores:0'),
							sess.graph.get_tensor_by_name('detection_boxes:0'),
							sess.graph.get_tensor_by_name('detection_classes:0')],
							feed_dict={'image_tensor:0':inp.reshape(1,inp.shape[0],inp.shape[1],3)})
	return (output,image)

def show_image(output,image):
	row = image.shape[0]
	col = image.shape[1]

	#running loop for all detections made by model
	num_detections=output[0][0]
	for i in range(int(num_detections)):
		classID = output[3][0][i]
		score = output[1][0][i]
		box = [float(v) for v in output[2][0][i]]
		if score>0.4:
			x=box[1]*col
			y=box[0]*row
			right=box[3]*col
			bottom=box[2]*row
			#drawing rectangle
			cv2.rectangle(image,(int(x),int(y)),(int(right),int(bottom)),(125,225,51),thickness=2)

		class_name="eye"
		pos = (int(x),int(y))
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_size = .7
		font_color = (125,225,51)
		line_width=2
		#inserting text
		cv2.putText(image,class_name,pos,font,font_size,font_color,line_width)
	image=cv2.resize(image,(600,600))
	cv2.imshow('asdfs',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
def main():
	args = argparse.ArgumentParser()
	args.add_argument('-i','--input_path')
	arguments = args.parse_args()
	if arguments.input_path == None:
		raise IOError("enter image_path")

	input_path = arguments.input_path
	graph_def = generate_graph()
	output,image = detection(graph_def,input_path)
	show_image(output,image)

if __name__ == '__main__':
	main()

