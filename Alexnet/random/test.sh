#!/bin/bash
export PYTHONPATH="../../caffe/python:$PYTHONPATH"
# test on VOC val 1449
datadir=/DATA/xhzhan
prefix=snapshot/seg_iter
iter=42000
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
        --batch_size 10 \
        --show 0 \
        --save_seg 1 \
        --score_name "upscore" \
        --gpu_id 0 \
        --out_root=${prefix}_${iter}
