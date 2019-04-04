import argparse
from food_detector.recognition import Recognition
import cv2
import os
import numpy as np 
import random
import torch
import time



def overlay_boxes(image,labels,boxes):
       

        for box in boxes:
            top_left,bottom_right = box[:2], box[2:]
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), (0,255,0), 1
            )
        return image

def overlay_class_names(image,labels,boxes):
        """
        Adds detected class names and scores in the positions defined by the 
        top-left corner of the predicted bounding box
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """

        template = "{}"
        for box,label in zip(boxes,labels):
            x, y = box[:2]
            s = template.format(label)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )
        return image

# test using todays labels
def test_today_label(imgpath,output_path):
    recog  = Recognition()
    #today_label=[13,15,27,41,57,61,63,68]
    today_label = [58,78,95,114,77,53,98,72]
    recog.set_model(today_label)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    out_file = os.path.join(output_path,"results.txt")
    f=open(out_file,'w')
    begin=time.time()
    model_time=0

    for img_file in os.listdir(imgpath):
        if img_file.endswith(".jpg"):
            img = cv2.imread(os.path.join(imgpath,img_file))
            model_begin=time.time()
            output = recog.get_results(img)
            model_time+=(time.time()-model_begin)
            s=""
            labels = []
            boxes = []
            for item in output:
                labels.append(item['label'])
                boxes.append([item["x1"],item["y1"],item["x2"],item["y2"]])
                s+="label:{},(x1,y1):({},{}),(x2,y2):({},{})|".format(item["label"],item["x1"],item["y1"],item["x2"],item["y2"])
            f.write(s+"\n")
            images = overlay_boxes(img,labels,boxes)
            images = overlay_class_names(images,labels,boxes)
            cv2.imwrite(os.path.join(output_path,img_file),images)
    end = time.time()
    print("total_time:{},average time {},model time {}".format(end-begin,(end-begin)/len(os.listdir(imgpath)),model_time/len(os.listdir(imgpath))))


def build_lib(label_path):
    recog = Recognition()
    for root,subs,files in os.walk(label_path):
        for sub in subs:
            label = int(sub)
            filename = os.listdir(os.path.join(root,sub))
            random.shuffle(filename)
            count=0
            for index,f in enumerate(filename):
                if count<7:
                    if f.endswith(".jpg"):

                        img = cv2.imread(os.path.join(root,sub,f))
                        h,w,_=img.shape
                        boxes=torch.tensor([[0,0,w,h],])
                        features = torch.transpose(recog.classifier.get_out_put(img,boxes),0,1)
                        recog.features.update_lib(features,label)
                        count+=1
    recog.features.save_lib()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Food Project module test')
    parser.add_argument("--imgpath",type=str,help="the input image path folder")
    parser.add_argument("--output",type=str,help="the test result ouput path")
    parser.add_argument("--label",type=str,help="the label path")
    parser.add_argument("--create_lib",action='store_true',help="the label path")

    args = parser.parse_args()
    imgpath=args.imgpath
    output = args.output
    label_path = args.label
    if args.create_lib:
        build_lib(label_path)

    test_today_label(imgpath,output)
            

    
    
