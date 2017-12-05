import os
import sys
import numpy as np
import cv2
import scipy.io as sio
import tester
import resize_uniform
import pdb

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def load_color_LUT_21(fn):
    contents = sio.loadmat(fn)
    return contents.values()[0][:,::-1]

def get_img_size(filename):
    dim_list = list(cv2.imread(filename).shape)[:2]
    if not len(dim_list) == 2:
        print('Could not determine size of image %s' % filename)
        sys.exit(1)
    return dim_list

def check_img_list(root_dir, list_filename, lazy_check=0):
    with open(list_filename, 'r') as infile:
        img_list = [line.strip() for line in infile.readlines() if len(line.strip()) > 0]
    print("Checking image list. Image list size: %d" % len(img_list))
    if not os.path.isfile(root_dir + img_list[0]):
        print("Image %s not found" % (root_dir + img_list[0]))
        sys.exit(1)
    base_size = get_img_size(root_dir + img_list[0])
    if not lazy_check:
        for i in range(len(img_list)):
            img_fn = root_dir + img_list[i]
            print('Checking image: %s' % img_list[i])
            if not os.path.isfile(img_fn):
                print('Image %s not found' % img_fn)
                sys.exit(1)
            img_size = get_img_size(img_fn)
            if not (base_size[0] == img_size[0] and base_size[1] == img_size[1]):
                print('Image size not consistant, images: %s vs. %s' % (img_fn, root_dir + img_list[0]))
                sys.exit(1)
    return base_size, img_list

def test_eval_seg(net, model, test_root, origin_root, gt_root, test_list, out_root, uniform_size, mean, batch_size, show=0, save_seg=0, save_img=0, save_prob=0, use_hyper=0, hyper_downsample_rate=1, hyper_centroids_name=None, score_name='score', start=0, end=-1, gpu=-1, f=None):

    # init
    OUT_PROB=False
    LUT = load_color_LUT_21(os.path.dirname(__file__) + '/VOC_color_LUT_21.mat')
    if save_seg:
        if not os.path.isdir(out_root + '/seg/'):
            os.makedirs(out_root + '/seg/')
        if not os.path.isdir(out_root + '/vlz/'):
            os.makedirs(out_root + '/vlz/')
    if save_prob:
        if not os.path.isdir(out_root + '/prob/'):
            os.makedirs(out_root + "/prob/")
        if not os.path.isdir(out_root + '/shape/'):
            os.makedirs(out_root + "/shape/")
        OUT_PROB=True
    if save_img:
        if not os.path.isdir(out_root + '/img/'):
            os.makedirs(out_root + "/img/")
        OUT_PROB=True
    with open(test_list, 'r') as infile:
        img_list = [line.strip() for line in infile.readlines() if len(line.strip()) > 0]
    if end==-1:
        img_list = img_list[start:]
    elif end <= len(img_list):
        img_list = img_list[start:end]
    else:
        raise Exception('end should not be larger than img_list length')
    base_size = [uniform_size, uniform_size]
    mean_map = np.tile(mean, [base_size[0], base_size[1], 1]) # H x W x C

    # centroids init for hyper-column
    if use_hyper:
        hyper_dsr = hyper_downsample_rate
        hyper_total = (uniform_size/hyper_dsr) * (uniform_size/hyper_dsr)
        hyper_yc = np.tile(np.arange(0, uniform_size, hyper_dsr).reshape((uniform_size/hyper_dsr, 1)), [1, uniform_size/hyper_dsr]).reshape((hyper_total, 1))
        hyper_xc = np.tile(np.arange(0, uniform_size, hyper_dsr).reshape((1, uniform_size/hyper_dsr)), [uniform_size/hyper_dsr, 1]).reshape((hyper_total, 1))
        hyper_centroids_blob = np.tile(np.hstack((hyper_yc, hyper_xc)), [batch_size, 1, 1])

    # caffe model init
    caffe_tester = tester.Tester(net, model, gpu)
    if use_hyper:
        if hyper_centroids_name in caffe_tester.blobs.keys():
            caffe_tester.blobs[hyper_centroids_name].data.flat = hyper_centroids_blob.flat
        else:
            raise Exception("Can not find the blob: %s" % hyper_centroids_name)

    # loop list
    hist = np.zeros((21, 21))
    for i in range(0, len(img_list), batch_size):
        if i % 100 == 0:
            print('Processing: %d/%d' % (i, len(img_list)))
        true_batch_size = min(batch_size, len(img_list) - i)
        batch_data = np.zeros((batch_size, 3, base_size[0], base_size[1]), dtype=np.float)
        for k in range(true_batch_size):
            if not os.path.isfile(test_root + img_list[i + k]):
                raise Exception('file not exist: %s' % (test_root + img_list[i + k]))
                sys.exit(1)
            img = cv2.imread(test_root + img_list[i + k]) # BGR, 0-255
            batch_data[k,...] = (resize_uniform.resize_pad_to_fit(img, base_size, pad_value=mean) - mean_map).transpose((2,0,1))
        inputs = {}
        inputs['data'] = batch_data
        qblobs = caffe_tester.predict(inputs, {}, [score_name])
        if use_hyper:
            if len(qblobs[score_name].shape) != 2:
                raise Exception("qblobs[score_name] should have 2 axis")
        for k in range(0, true_batch_size):
            origin_img = cv2.imread(origin_root + img_list[i + k])
            origin_shape = origin_img.shape
            if gt_root is not None:
                gt_img = cv2.imread(gt_root + img_list[i + k].split('.')[0] + '.png')[:,:,0]
            if use_hyper:
                if OUT_PROB:
                    prob_map = qblobs[score_name].reshape((batch_size, uniform_size/hyper_dsr, uniform_size/hyper_dsr,-1))[k,...].transpose((2,0,1))
                cls_map = qblobs[score_name].argmax(axis=1).reshape((batch_size, uniform_size/hyper_dsr,uniform_size/hyper_dsr))[k,...].astype(np.uint8)
            else:
                if OUT_PROB:
                    prob_map = qblobs[score_name][k]
                cls_map = np.array(qblobs[score_name][k].transpose(1, 2, 0).argmax(axis=2), dtype=np.uint8)
            if save_prob:
                np.save(out_root + "/prob/" + img_list[i+k].split('.')[0] + ".npy", prob_map)
                np.save(out_root + "/shape/" + img_list[i+k].split('.')[0] + ".npy", origin_shape[:2])
            if save_img:
                cv2.imwrite(out_root + '/img/' + img_list[i + k], origin_img)

            # origin size
            out_map = np.uint8(LUT[cls_map] * 255)
            cls_map_origin = resize_uniform.resize_crop_to_fit(cls_map, origin_shape[:2], interp=cv2.INTER_NEAREST)
            out_map_origin = resize_uniform.resize_crop_to_fit(out_map, origin_shape[:2], interp=cv2.INTER_NEAREST)
            
            # mIU
            if gt_root is not None:
                hist += fast_hist(gt_img.flatten(), cls_map_origin.flatten(), 21)
            if show:
                cv2.imshow("image", origin_img)
                cv2.imshow("seg result", out_map_origin)
                cv2.waitKey(0)
            if save_seg:
                cls_map_fn = out_root + '/seg/' + img_list[i + k].split('.')[0] + ".png"
                out_map_fn = out_root + "/vlz/" + img_list[i + k].split('.')[0] + ".png"
                if not os.path.isdir(os.path.dirname(cls_map_fn)):
                    os.makedirs(os.path.dirname(cls_map_fn))
                if not os.path.isdir(os.path.dirname(out_map_fn)):
                    os.makedirs(os.path.dirname(out_map_fn))
                cv2.imwrite(cls_map_fn, cls_map_origin)
                cv2.imwrite(out_map_fn, out_map_origin)
    # results
    if gt_root is not None:
        acc = np.diag(hist).sum() / hist.sum()
        print '>>>', 'overall accuracy', acc
        acc = np.diag(hist) / hist.sum(1)
        print '>>>', 'mean accuracy', np.nanmean(acc)
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        print '>>>', 'per class IU:\n', iu
        show_iu = ["{:.2f}".format(i*100) for i in iu]
        print '>>>', show_iu
        print '>>>', 'mean IU', np.nanmean(iu)
        if f is not None:
            f.write('model: %s\n' % model)
            f.write('%s\n' % show_iu)
            f.write('Mean IU: %f\n\n' % np.nanmean(iu))


def main_eval():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('net')   
    parser.add_argument('model')
    parser.add_argument('test_root')
    parser.add_argument('test_list')
    parser.add_argument('--size', type=int)
    parser.add_argument('--mean', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--gt_root', type=str, default=None)
    parser.add_argument('--out_root', type=str, default=None)
    parser.add_argument('--show', type=int, default=0)
    parser.add_argument('--save_seg', type=int, default=0)
    parser.add_argument('--save_img', type=int, default=0)
    parser.add_argument('--save_prob', type=int, default=0)
    parser.add_argument('--output_fn', type=str, default=None)
    parser.add_argument('--use_hyper', type=int, default=0)
    parser.add_argument('--hyper_downsample_rate', type=int, default=1)
    parser.add_argument('--hyper_centroids_name', type=str, default=None)
    parser.add_argument('--score_name', type=str, default=None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1) # -1 means to the end of the list
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    if args.out_root is not None:
        if not os.path.isdir(args.out_root):
            os.makedirs(args.out_root)
    if args.mean == 'imgnet_mean':
        mean = np.array([104.00699, 116.66877, 122.67892])
    elif args.mean == 'zero_mean':
        mean = np.array([0.0, 0.0, 0.0])
    elif args.mean == 'jigsaw_mean':
        mean = np.array([104.0, 117.0, 123.0])
    else:
        print('Unknown mean value type: %s. Existing mean types: imgnet_mean, zero_mean, jigsaw_mean' % args.mean)
        sys.exit(1)
    origin_root = args.test_root + '/'
    if args.output_fn is not None:
        f = open(args.output_fn, 'a')
    elif args.out_root is not None and args.gt_root is not None:
        f = open(args.out_root + "/results.txt", 'a')
    else:
        f = None
    test_eval_seg(args.net, args.model, args.test_root, origin_root, args.gt_root, args.test_list, args.out_root, args.size, mean, args.batch_size, args.show, args.save_seg, args.save_img, args.save_prob, args.use_hyper, args.hyper_downsample_rate, args.hyper_centroids_name, args.score_name, args.start, args.end, args.gpu_id, f)
    if f is not None:
        f.close()

if __name__ == "__main__":
    main_eval()
