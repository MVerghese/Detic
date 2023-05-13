# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import mss

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo
import nltk
from PIL import Image
import cv2
import torch
import clip

# Fake a video capture object OpenCV style - half width, half height of first screen using MSS
class ScreenGrab:
    def __init__(self):
        self.sct = mss.mss()
        m0 = self.sct.monitors[0]
        self.monitor = {'top': 0, 'left': 0, 'width': m0['width'] / 2, 'height': m0['height'] / 2}

    def read(self):
        img =  np.array(self.sct.grab(self.monitor))
        nf = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return (True, nf)

    def isOpened(self):
        return True
    def release(self):
        return True


# constants
WINDOW_NAME = "Detic"

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def expand_bbox(bbox,scale):
    bbox = bbox.reshape(4)
    dx = bbox[2]-bbox[0]
    dy = bbox[3]-bbox[1]
    scalex, scaley = dx*(scale-1)/2, dy*(scale-1)/2
    new_bbox = np.array([[max(0,bbox[0]-scalex),max(0,bbox[1]-scaley),min(640,bbox[2]+scalex),min(480,bbox[3]+scaley)]])
    return(new_bbox)

class DeticModule:

    def __init__(self,obj_lang_tags,output_score_threshold=0.3,load_clip=False):
        noun_list = ", ".join(obj_lang_tags)
        noun_list = '"'+noun_list+'"'
        mp.set_start_method("spawn", force=True)
        args = get_parser().parse_args(['--config-file', '/home/mverghese/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml', '--input', '"test.jpg, desk.jpg"' ,'--vocabulary', 'custom', '--custom_vocabulary', noun_list,  '--opts', 'MODEL.WEIGHTS', '/home/mverghese/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'])
        setup_logger(name="fvcore")
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)

        self.demo = VisualizationDemo(cfg, args)
        for cascade_stages in range(len(self.demo.predictor.model.roi_heads.box_predictor)):
                self.demo.predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold

        if load_clip:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {text}") for text in obj_lang_tags]).to(self. device)
            with torch.no_grad():
                self.text_features = self.model.encode_text(text_inputs)
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True)


    def run_detection(self,img, visualize=False, vis_length=0):

        predictions, visualized_output = self.demo.run_on_image(img)

        if visualize:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            cv2.waitKey(vis_length)
                # break  # esc to quit

        return(predictions)

    def get_image_crop(self,img,crop_buffer = 1.3):
        predictions = self.run_detection(img)
        bboxes = predictions['instances'].pred_boxes.tensor.cpu().numpy()
        if bboxes.shape[0] == 0:
            return(img)
        total_bbox = np.array([np.concatenate((np.min(bboxes[:,:2],axis=0),np.max(bboxes[:,2:],axis=0)))])
        total_bbox = expand_bbox(total_bbox,crop_buffer)
        total_bbox = total_bbox.reshape(4).astype(int)
        cropped_image = img[total_bbox[1]:total_bbox[3],total_bbox[0]:total_bbox[2],:]
        # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        # im = Image.fromarray(cropped_image)
        return(cropped_image)

    def get_clip_embeddings(self,img):
        image = Image.fromarray(img)

        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        return(image_features)

    def get_clip_probs(self,img):
        image_features = self.get_clip_embeddings(img)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        similarity = similarity.cpu().numpy().flatten()
        return(similarity)











if __name__ == "__main__":

    detic_interface = DeticModule(["pan","spatula"],output_score_threshold=0.3)
    img = read_image('PlayKitchen2_Oven.jpg', format="BGR")
    predictions = detic_interface.run_detection(img,visualize=True,vis_length=0)
    print(predictions['instances'].scores.shape)
    print(predictions)


    # list_image_names = os.listdir('training_images/') 
    # list_txt = [x[:-6].replace('_',' ') for x in list_image_names]
    # print(list_txt)
    # list_image_path = ['training_images/'+x for x in list_image_names]
    # print(list_image_path)

    # nouns = []
    # for text in list_txt:
    #     is_noun = lambda pos: pos[:2] == 'NN'
    #     # do the nlp stuff
    #     tokenized = nltk.word_tokenize(text)
    #     new_nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
    #     nouns = nouns + new_nouns
    # nouns = list(set(nouns))
    # print("NOUNS")
    # print(nouns)

    # noun_list = ", ".join(nouns)
    # noun_list = '"'+noun_list+'"'
    # print(noun_list)

    # mp.set_start_method("spawn", force=True)
    # args = get_parser().parse_args(['--config-file', '/home/mverghese/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml', '--input', '"test.jpg, desk.jpg"' ,'--vocabulary', 'custom', '--custom_vocabulary', noun_list,  '--opts', 'MODEL.WEIGHTS', '/home/mverghese/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'])
    # setup_logger(name="fvcore")
    # logger = setup_logger()
    # logger.info("Arguments: " + str(args))

    # cfg = setup_cfg(args)

    # demo = VisualizationDemo(cfg, args)


    # for idx in range(len(list_image_path)):
    #     path = list_image_path[idx]
    #     img = read_image(path, format="RGB")
    #     print(img.shape)
    #     start_time = time.time()
    #     predictions, visualized_output = demo.run_on_image(img)
    #     bboxes = predictions['instances'].pred_boxes.tensor.cpu().numpy()
    #     print(bboxes)
    #     total_bbox = np.array([np.concatenate((np.min(bboxes[:,:2],axis=0),np.max(bboxes[:,2:],axis=0)))])
    #     total_bbox = expand_bbox(total_bbox,1.3)
    #     total_bbox = total_bbox.reshape(4).astype(int)
    #     print(total_bbox)
    #     cropped_image = img[total_bbox[1]:total_bbox[3],total_bbox[0]:total_bbox[2],:]
    #     cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    #     im = Image.fromarray(cropped_image)
    #     im.save('cropped_training_images/'+list_image_names[idx])
    #     # scaled_boxes = predictions['instances'].pred_boxes.clone()
    #     # scaled_boxes.scale(1.2,1.2)
    #     # print(scaled_boxes)
    #     logger.info(
    #         "{}: {} in {:.2f}s".format(
    #             path,
    #             "detected {} instances".format(len(predictions["instances"]))
    #             if "instances" in predictions
    #             else "finished",
    #             time.time() - start_time,
    #         )
    #     )

        # if args.output:
        #     if os.path.isdir(args.output):
        #         assert os.path.isdir(args.output), args.output
        #         out_filename = os.path.join(args.output, os.path.basename(path))
        #     else:
        #         assert len(args.input) == 1, "Please specify a directory with args.output"
        #         out_filename = args.output
        #     visualized_output.save(out_filename)
        # else:
        #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
        #     if cv2.waitKey(0) == 27:
        #         break  # esc to quit
    