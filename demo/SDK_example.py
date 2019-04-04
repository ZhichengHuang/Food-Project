import argparse
from food_detector.recognition import Recognition
import cv2
import os
import numpy as np 
import random
import torch
import time



# test using todays labels
def test_today_label():
    recog  = Recognition()
    #today_label=[13,15,27,41,57,61,63,68]
    today_label = [58,78,95,114,77,53,98,72]
    flag=recog.set_model(today_label)
    if flag==0:
        print("error")
    else:
        print("set successfully!")
    img = cv2.imread("test.jpg")
    output = recog.get_results(img)
    print(output)


def register_dish():
    recog = Recognition()
    label=1
    img = cv2.imread("test.jpg")
    out=recog.registe_model(img,label)
    print(out)

if __name__=="__main__":
    test_today_label()

    # regiseter dishes
    register_dish()

            

    
    
