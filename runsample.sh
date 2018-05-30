#!/bin/bash

xhost +

GPU=0 docker run --runtime=nvidia --privileged --rm -it --env DISPLAY=$DISPLAY --env="QT_X11_NO_MITSHM=1" -v /dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix:ro --name gtc2018 gtc2018:0.3 python2 /detectron/tools/infer_simple_show.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml --output-dir /detectron/demo/output --image-ext jpg --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl demo

xhost -
