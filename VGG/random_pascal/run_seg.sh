export PYTHONPATH="../../resources:$PYTHONPATH"

CAFFE=../../caffe/cmake_build/install/bin/caffe
snapshotdir=snapshot
while [ ! -e $snapshotdir/graph_iter_8000.caffemodel ]; do
    echo "Graph training not finished. Waiting ..."
    sleep 1m
done
mpirun -np 4 $CAFFE train --solver=solver_seg.prototxt --weights=$snapshotdir/graph_iter_8000.caffemodel 2>&1 |tee log/finetune_seg.log
