from maskrcnn_benchmark.config import cfg
import os

import torch

from .dishes_detecor import Dishes
from .dishes_classifier import feature_extractor
from .ground_feature import feature_lib

class Recognition:
    def __init__(self,configfile=os.path.join(os.path.dirname(__file__),"config.yaml")):
        cfg.merge_from_file(configfile)

        self.features = feature_lib(cfg)
        self.dish_detector = Dishes(
            cfg,
            min_image_size=720,
            confidence_threshold=0.5
        )
        self.classifier = feature_extractor(cfg)

    def get_results(self,img):
        if img is None:
            return "Image error"
        output=[]
        boxes = self.dish_detector.run_on_opencv_image(img)
        if len(boxes)<1:
            tmp={
                "label": 0,
                "score": 0,
                "x1": 0,
                "y1": 0, 
                "x2": 0,
                "y2": 0
            }
            output.append(tmp)
            return output
        else:
            features = self.classifier.get_out_put(img,boxes)
            labels = self.features.get_label(features)
            
            for box,label in zip(boxes,labels):
                box = box.to(torch.int64)
                box_list = box.tolist()
                tmp={
                    "label": int(label),
                    "score": 1,
                    "x1": box_list[0],
                    "y1": box_list[1], 
                    "x2": box_list[2],
                    "y2": box_list[3]
                }
                output.append(tmp)
            return output
    
    
    def registe_model(self,img,label):
        """
        img: the input image (BGR)
        label: int 
        """
        output=[]
        if img is None:
            tmp={
                "label": 0,
                "score": 0,
                "x1": 0,
                "y1": 0, 
                "x2": 0,
                "y2": 0
            }
            output.append(tmp)
            return output
        
        boxes = self.dish_detector.run_on_opencv_image(img)
        if len(boxes)<1:
            tmp={
                "label": 0,
                "score": 0,
                "x1": 0,
                "y1": 0, 
                "x2": 0,
                "y2": 0
            }
            output.append(tmp)
            return output
        else:
            try:

                features = torch.transpose(self.classifier.get_out_put(img,boxes),0,1)
                self.features.update_lib(features,label)
                self.features.fix_lib()
                
                for box,label in zip(boxes,[label,]):
                    box = box.to(torch.int64)
                    box_list = box.tolist()
                    tmp={
                        "label": int(label),
                        "score": 1,
                        "x1": box_list[0],
                        "y1": box_list[1], 
                        "x2": box_list[2],
                        "y2": box_list[3]
                    }
                    output.append(tmp)
                    self.features.save_lib()
            except Exception:
                tmp={
                "label": 0,
                "score": 0,
                "x1": 0,
                "y1": 0, 
                "x2": 0,
                "y2": 0
                }
                output.append(tmp)
                return output
            else:

                return output
    
    def set_model(self,labels):
        """
        labels: list, todays label list each element is int 
        return : 0 (error), 1 (successful)
        """
        return self.features.get_today_lib(labels)




