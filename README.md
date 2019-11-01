# Implementation of Mix-and-Match Tuning for Self-Supervised Semantic Segmentation.

### Paper

Xiaohang Zhan, Ziwei Liu, Ping Luo, Xiaoou Tang, Chen Change Loy, ["Mix-and-Match Tuning for Self-Supervised Semantic Segmentation"](https://arxiv.org/abs/1712.00661), AAAI 2018

Project Page:
[link](http://mmlab.ie.cuhk.edu.hk/projects/M&M/)

### Dependency
Library (Note that the versions are not strictly restricted):
    OpenMPI=1.8.5, CUDA=8.0, CUDNN=5.1.10

Python:
    cv2

### Before Start
1. Download pre-trained models in [link](https://drive.google.com/drive/folders/1dAA1aWTll_GAgpgYdYJnhrqY2pEydWiv?usp=sharing) to `pretrain`. (You can also find trained `graph_iter_xxx.caffemodel` which can directly be fine-tuned for segmentation.)

2. Download PASCAL VOC 2012 augmented dataset and CityScapes dataset to a proper position.

    For PASCAL VOC 2012, create standard training list as shown in `data/pascal/train.txt` and validation list as shown in `data/pascal/val.txt`

    For CityScapes, create standard training list as shown in `data/cityscapes/train.txt` and validation list as shown in `data/cityscapes/val.txt`

3. Build caffe with cmake
    ```
    cd caffe
    sh build.sh
    ```

### training
For example, train alexnet with colorization as pretrained model.
```
cd Alexnet/colorize
```
Then edit `train_graph.prototxt` and `finetune_seg.prototxt` to specify "source" and "root_dir" in
the data layer.
```
sh run_graph.sh # or use trained models in the Google Drive folder.
sh run_seg.sh
```

### testing
Edit test.sh to specify data root, testing list and ground truth root.
```
sh test.sh
```
Testing results are saved in `snapshot/seg_iter_xxx/` by default.

### Citing Mix-and-Match
```
@inproceedings{zhan2018mix,
 author = {Zhan, Xiaohang and Liu, Ziwei and Luo, Ping and Tang, Xiaoou and Loy, Chen Change},
 title = {Mix-and-Match Tuning for Self-Supervised Semantic Segmentation},
 booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
 month = {February},
 year = {2018} 
}
```

### Acknowledgement
The CAFFE is forked from [https://github.com/yjxiong/caffe](https://github.com/yjxiong/caffe)
