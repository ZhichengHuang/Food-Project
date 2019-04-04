import cv2
import os
import torch
from torchvision import transforms as T 

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
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
    
    def run_on_opencv_image(self,image):
        """
        Arguments:
            Image (np.ndarrat) an image as returned by OpenCV
        Returns:
            prediction (BoxList): the detected objects Additional information
            of the detection properties can be found in the fields of the BoxList via prediction.dields()
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)
        return top_predictions.bbox

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
        # convert to an ImageList, padded so that it is divisible by cfg.DATALOADERR.SIZE_DIVISIBILITY
        image_list= to_image_list(image,self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single imsge is passed at a time
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
    
    



