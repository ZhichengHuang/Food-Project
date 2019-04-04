import torch
import os

from collections import Counter




class feature_lib:
    def __init__(self,cfg):
        self.feature_lib_path= cfg.FEATURELIB.PATH
        self.lib = self.load_lib()
        # size 2048*n
        self.lib_feature = self.lib['feature']
        self.lib_label = self.lib['labels']
        # 白米饭,黑米饭,小碗汤,西瓜,红薯,玉米,馒头,粥,花卷,酸奶,小菜
        self.common_labels=cfg.FEATURELIB.COMMONLABEL
    
    def save_lib(self):
        lib={
            "feature":self.lib_feature,
            "labels": self.lib_label
        }
        torch.save(lib,self.feature_lib_path)
        

    
    def fix_lib(self):
        self.today_label=self.lib_label
        self.today_lib = self.lib_feature


    def load_lib(self):
        if os.path.exists(self.feature_lib_path):
            lib = torch.load(self.feature_lib_path)
            return lib
        else:
            lib={"feature":None,
                "labels":[]}
            return lib
        
    def get_today_lib(self,labels):
        try:

            labels.extend(self.common_labels)
            today_feature_list=[]
            today_labels=[]
            for index,la in enumerate(self.lib_label):
                if la in labels:
                    f = self.lib_feature[:,index]
                    today_feature_list.append(f)
                    today_labels.append(la)
            self.today_lib = torch.stack(today_feature_list,dim=1)
            self.today_label = today_labels
        except Exception:
            return 0
        else:
            return 1
            
        

    def update_lib(self,feature,label):
        print("feature=",feature.size())
        if self.lib_feature is None:
            self.lib_feature = feature
            self.lib_label.append(label)
        else:
            self.lib_feature = torch.cat([self.lib_feature,feature],dim=1)
            self.lib_label.append(label)


    
    
    def get_label(self,in_feature,th=0.85):
        """
        return : -1 stands for the unrecognition dishes
        """
        sim = torch.matmul(in_feature,self.today_lib)
        mask = sim>th
        mask_list = mask.tolist()
        result_list=[]
        for mask_item in mask_list:
            out_label = [label for label,index in zip( self.today_label,mask_item) if index==1]
            if len(out_label)>0:
                out = Counter(out_label).most_common(1)[0][0]
                result_list.append(out)
            else:
                result_list.append(0)
        return result_list
    


    