import cv2
import numpy as np
#####################LOAD NETWORK
net=cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
classes=[]
with open ('coco.names','r') as f:
	classes=[line.strip() for line in f.readlines()]
# print (classes)
# ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
layer_names=net.getLayerNames()
output_layers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

colors=np.random.uniform(0,255,(len(classes),3))
#####################LOAD IMAGE
img=cv2.imread("test1.jpg")
img=cv2.resize(img,None,fx=0.1,fy=0.1)
height,width,channels=img.shape
blob=cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)

# for b in blob:
# 	n=0
# 	for img_blob in b:
# 		n=n+1
# 		cv2.imshow(str(n),img_blob)

net.setInput(blob)
outs=net.forward(output_layers)
# print(outs)

###################EXTRACTING OBJECTS
class_ids=[]
confidences=[]
boxes=[]
for out in outs:
	for detection in out:
		scores=detection[5:]
		class_id=np.argmax(scores)
		confidence=scores[class_id]
		if confidence>0.5:
			centre_x=int(detection[0]*width)
			centre_y=int(detection[1]*height)
			w=int(detection[2]*width)
			h=int(detection[3]*height)
			# cv2.circle(img,(centre_x,centre_y),10,(0,255,0),2)
			x=int(centre_x-w/2)
			y=int(centre_y-h/2)
			# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			# cv2.imshow("",img)
			boxes.append([x,y,w,h])
			confidences.append(float(confidence))
			class_ids.append(class_id)
number_of_objects=len(boxes)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(number_of_objects)
print(indexes)
for i in range (len(boxes)):
	if i in indexes:
		x,y,w,h=boxes[i]
		label=str(classes[class_ids[i]])
		font=cv2.FONT_HERSHEY_PLAIN
		color=colors[i]
		cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
		cv2.putText(img,label,(x,y+3),font,2,color,3)



cv2.imshow("",img)

cv2.waitKey(0)
cv2.destroyAllWindows()