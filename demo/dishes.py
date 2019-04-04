import cv2
import os
import torch
from torchvision import transforms as T 

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L 
from maskrcnn_benchmark.utils import cv2_util

class Dishes(object):
    CATEGORIES = [
        "__background",
        "1"]
    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=720,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device= torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir =  cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg,self.model,save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    
    def build_transform(self):
        """
        Creates a basic transformation that was used to train the model
        """
        cfg = self.cfg
        """
        we are Loading images with Opencv, so we don't need to convert them to BGR
        they are already! So all we need to do is to normalize by 255 if we want to convert to BGR255 format, or flip the channels
        if we want it to be in BGR in [0-1]
        """
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])
        
        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self,image,name,out_path,is_crop=False):
        """
        Arguments:
            Image (np.ndarrat) an image as returned by OpenCV
        Returns:
            prediction (BoxList): the detected objects Additional information
            of the detection properties can be found in the fields of the BoxList via prediction.dields()
        """

        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        result = self.overlay_boxes(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)
        if is_crop:
            self.crop_boxes(image,top_predictions,name,out_path)


        return result

    def compute_prediction(self,original_image):
        """
        Arguments:
            original_image (np.ndarray) : an image as returned by OpenCV
        Returns:
            prediction(BoxList): the detected objects. Addition information of the detection properties
            can be found in the fields of the BoxList via prediction.fields()
        """

        # apply preprocessing to image
        image = self.transforms(original_image)
        #convert to an ImageList, padded so that it is divisible by cfg.DATALOADERR.SIZE_DIVISIBILITY
        image_list= to_image_list(image,self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]
        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        return prediction

    def select_top_predictions(self,predictions):
        """
        Select only predictions with have a 'socre' > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model
                It should contain the field 'scores'.
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field('scores')
        _,idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors
    
    def overlay_boxes(self,image,predictions):
        """
        Adds the predicted boxes on top of the image
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box,color in zip(boxes,colors):
            box = box.to(torch.int64)
            top_left,bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )
        return image
    def overlay_class_names(self,image,predictions):
        """
        Adds detected class names and scores in the positions defined by the 
        top-left corner of the predicted bounding box
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box,score,label in zip(boxes,scores,labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )
        return image
    def crop_boxes(self,image,predictions,name,out_path):
        boxes = predictions.bbox
        for index,box in enumerate(boxes):
            box = box.to(torch.int64)
            top_left,bottom_right = box[:2].tolist(), box[2:].tolist()
            roi = image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
            cv2.imwrite(os.path.join(out_path,name+"_"+str(index)+".jpg"),roi)




