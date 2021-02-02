import cv2
import numpy as np
import Calculations

def loadNet():
    net = cv2.dnn.readNet("Resources/converted.weights","Resources/yolov3.cfg")
    classes = ["Vehicles"]
    layer_names = net.getLayerNames()
    colors = (255,0,0)
    outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    return net, classes, colors, outputlayers

def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416,416),mean=(0,0,0), swapRB=True, crop = False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def labels(outputs, height, width, img):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	cv2.imshow("Image", img)

def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(path, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels

def imageDetection(path):
    model, classes, colors, outputLayers = loadNet()
    image,height,width,channels = load_image(path)
    blob, outputs = detect_objects(image,model,outputLayers)
    labels(outputs,height,width,image)
    while True:
        if cv2.waitKey(1) == 27:
            break

def videoDetection(path):
    model, classes, colors, output_layers = loadNet()
    cap = cv2.VideoCapture(path)
    while True:
        _, frame = cap.read()
        height,width,channels = frame.shape
        blob,outputs = detect_objects(frame,model,output_layers)
        labels(outputs, height, width, frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == "__main__":
    videoDetection('Resources/cars.mp4')


