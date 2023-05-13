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



if __name__ == "__main__":

    list_image_names = os.listdir('val_images/') 
    list_txt = [x[:-6].replace('_',' ') for x in list_image_names]
    print(list_txt)
    list_image_path = ['val_images/'+x for x in list_image_names]
    print(list_image_path)

    nouns = []
    for text in list_txt:
        is_noun = lambda pos: pos[:2] == 'NN'
        # do the nlp stuff
        tokenized = nltk.word_tokenize(text)
        new_nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        nouns = nouns + new_nouns
    nouns = list(set(nouns))
    nouns.append("oven")
    print("NOUNS")
    print(nouns)

    noun_list = ", ".join(nouns)
    noun_list = '"'+noun_list+'"'
    print(noun_list)

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args(['--config-file', 'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml', '--input', '"test.jpg, desk.jpg"' ,'--vocabulary', 'custom', '--custom_vocabulary', noun_list,  '--opts', 'MODEL.WEIGHTS', 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'])
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, args)


    for idx in range(len(list_image_path)):
    # for idx in range(list_image_path.index("training_images/mustard_on_counter_049.jpg"),list_image_path.index("training_images/mustard_on_counter_049.jpg")+1):
        path = list_image_path[idx]
        img = read_image(path, format="BGR")
        print(img.shape)
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        lang_classes = []
        for noun in nouns:
            if noun in list_txt[idx]:
                lang_classes.append(nouns.index(noun))

        # import pdb; pdb.set_trace()
        pred_classes = predictions['instances'].pred_classes.cpu().numpy()

        bbox_idxs = np.in1d(pred_classes,lang_classes).nonzero()[0]

        # print(predictions['instances'].scores.cpu().numpy())
        bboxes = predictions['instances'].pred_boxes.tensor.cpu().numpy()
        print(bboxes)
        lang_bboxes = bboxes[bbox_idxs,:]
        if lang_bboxes.shape[0] == 0:
            print("Object not found in image")
            continue
        print(lang_bboxes)
        total_bbox = np.array([np.concatenate((np.min(lang_bboxes[:,:2],axis=0),np.max(lang_bboxes[:,2:],axis=0)))])
        total_bbox = expand_bbox(total_bbox,1.5)
        total_bbox = total_bbox.reshape(4).astype(int)
        print(total_bbox)
        cropped_image = img[total_bbox[1]:total_bbox[3],total_bbox[0]:total_bbox[2],:]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(cropped_image)
        im.save('cropped_val_images/'+list_image_names[idx])
        # scaled_boxes = predictions['instances'].pred_boxes.clone()
        # scaled_boxes.scale(1.2,1.2)
        # print(scaled_boxes)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

    #     if args.output:
    #         if os.path.isdir(args.output):
    #             assert os.path.isdir(args.output), args.output
    #             out_filename = os.path.join(args.output, os.path.basename(path))
    #         else:
    #             assert len(args.input) == 1, "Please specify a directory with args.output"
    #             out_filename = args.output
    #         visualized_output.save(out_filename)
    #     else:
    #         cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #         cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
    #         if cv2.waitKey(0) == 27:
    #             break  # esc to quit
    # 