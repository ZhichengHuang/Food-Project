import os
from maskrcnn_benchmark.config import cfg
from dishes import Dishes
import cv2





def init_function(config_file,images,out_path,crop_path=None,is_crop=False):
    cfg.merge_from_file(config_file)
    # manual override some options
    # cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

    dish = Dishes(
        cfg,
        min_image_size=720,
        confidence_threshold=0.85
    )
    for item in os.listdir(images):
        if item.endswith(".jpg"):
            img = cv2.imread(os.path.join(images,item))
            predictions = dish.run_on_opencv_image(img,item,crop_path,is_crop=is_crop)
            cv2.imwrite(os.path.join(out_path,item),predictions)

def init_function_by_date(config_file,images,out_path,crop_path=None,is_crop=True):
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.WEIGHT", "/data/home/v-zhihua/Food_search/maskrcnn-benchmark/saved_model/model/best_online.pth.tar"])
    dish = Dishes(
        cfg,
        min_image_size=720,
        confidence_threshold=0.5
    )
    for root,subs,files in os.walk(images):
        for sub in subs:
            for file in os.listdir(os.path.join(root,sub)):
                if file.endswith(".jpg"):
                    img = cv2.imread(os.path.join(root,sub,file))
                    crop_ = os.path.join(crop_path,sub)
                    if not os.path.exists(crop_):
                        os.makedirs(crop_)
                    predictions = dish.run_on_opencv_image(img,file,crop_,is_crop=is_crop)


if __name__=="__main__":
    config_file="/data/home/v-zhihua/Food_search/model_config_bbs/config/food_4gpu.yaml"
    images="/data/home/v-zhihua/Food_search/test_data/rename"
    out_path="/data/home/v-zhihua/Food_search/test_data/out"
    crop_path="/data/home/v-zhihua/Food_search/test_data/crop"
    # init_function(config_file,images,out_path,crop_path)
    init_function_by_date(config_file,images,out_path,crop_path)


