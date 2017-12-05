#!/bin/bash
export PYTHONPATH="../../caffe/python:$PYTHONPATH"
# test on VOC val 1449
#datadir=/PATH/TO/YOUR/DATA/ROOT
datadir=/DATA/xhzhan
prefix=snapshot/seg_iter
iter=42000
dsr=2 # downsample rate
while [ ! -f ${prefix}_${iter}.caffemodel ]; do
    sleep 1m
done
python ../../resources/test_pascal.py \
        deploy.prototxt \
        ${prefix}_${iter}.caffemodel \
        $datadir/VOCdevkit/VOC_aug/JPEGImages/ \
        $datadir/VOCdevkit/VOC_aug/Lists/Img/val.txt \
        --gt_root=$datadir/VOCdevkit/VOC_aug/SegmentationClass_label/ \
        --size 512 \
        --mean imgnet_mean \
        --batch_size 1 \
        --show 0 \
        --save_seg 1 \
        --use_hyper 1 \
        --hyper_downsample_rate $dsr \
        --hyper_centroids_name "centroids_float" \
        --score_name "score_column" \
        --gpu_id 4 \
        --out_root=${prefix}_${iter}
