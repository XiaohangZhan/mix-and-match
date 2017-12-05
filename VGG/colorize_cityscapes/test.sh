#!/bin/bash
export PYTHONPATH="../../caffe/python:$PYTHONPATH"
# test on VOC val 1449
#datadir=/PATH/TO/YOUR/DATA/ROOT
datadir=/home/xhzhan/data
prefix=snapshot/seg_iter
iter=42000
dsr=2 # downsample rate
while [ ! -f ${prefix}_${iter}.caffemodel ]; do
    sleep 1m
done
python ../../resources/test_cityscapes.py \
        deploy.prototxt \
        ${prefix}_${iter}.caffemodel \
        $datadir/CityScapes \
        $datadir/CityScapes/trainlist/val.txt \
        --gt_root=$datadir/CityScapes/ \
        --crop_size 512 \
        --stride_ratio 0.5 \
        --mean zero_mean \
        --batch_size 1 \
        --show 0 \
        --save_seg 1 \
        --use_hyper 1 \
        --hyper_downsample_rate $dsr \
        --hyper_centroids_name "centroids_float" \
        --score_name "softmax_score" \
        --gpu_id 4 \
        --out_root=${prefix}_${iter}
