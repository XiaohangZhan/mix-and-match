export PYTHONPATH="../../resources:$PYTHONPATH"
export PYTHONPATH="../../caffe/python:$PYTHONPATH"
CAFFE=../../caffe/cmake_build/install/bin/caffe

if [ ! -d log ]; then
    mkdir log
fi
if [ ! -d snapshot ]; then
    mkdir snapshot
fi
if [ ! -f ../../caffe/cmake_build/install/bin/caffe ]; then
    echo Caffe executable file not found. Please cmake caffe at first.
    exit
fi
if [ ! -f ../../pretrain/alexnet_jigsaw.caffemodel ]; then
    echo pre-trained file not found. Please execute download_pretrain.sh.
    exit
fi

$CAFFE train --solver=solver_graph.prototxt --weights ../../pretrain/alexnet_jigsaw.caffemodel 2>&1 |tee log/graph.log
