# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2, os

from fcos_core.config import cfg
from predictor import COCODemo
import sys
import time




def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../coco_psp_aug.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        default="/home/kyle/Programs/FCOS/FCOS/training_dir/fcos_imprv_R_50_FPN_1x/model_final.pth",
        metavar="FILE",
        help="path to the trained model",
    )
    parser.add_argument(
        "--images-dir",
        default="demo/images",
        metavar="DIR",
        help="path to demo images directory",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights

    cfg.freeze()



    thresholds_for_classes = [0.5761297345161438]

    demo_im_names = os.listdir(args.images_dir)
    
    candidate = open(args.images_dir+"candidate.txt","w+")
    none = open(args.images_dir+"none.txt","w+")
    os.mkdir(args.images_dir+"Labeled/")
    
    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_thresholds_for_classes=thresholds_for_classes,
        min_image_size=args.min_image_size
    )

    for im_name in demo_im_names:
        img = cv2.imread(os.path.join(args.images_dir, im_name))
        if img is None:
            continue
        start_time = time.time()
        result = coco_demo.judge_image(img)
        print(im_name,result)
        if result:
            candidate.write(im_name+"\n")
            composite = coco_demo.run_on_opencv_image(img)
            cv2.imwrite(args.images_dir+"Labeled/"+im_name, composite)
        else:
            none.write(im_name+"\n")
    candidate.close()
    none.close()

if __name__ == "__main__":
    main()

