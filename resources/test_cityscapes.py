import os
import sys
import numpy as np
import cv2
import scipy.io as sio
import tester
import crop_images

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def load_color_LUT(fn):
    contents = sio.loadmat(fn)
    return contents.values()[0][:,::-1]

def get_img_size(filename):
    dim_list = list(cv2.imread(filename).shape)[:2]
    if not len(dim_list) == 2:
        print('Could not determine size of image %s' % filename)
        sys.exit(1)
    return dim_list

def check_img_list(root_dir, list_filename, lazy_check = 0):
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

def test_eval_seg(net, model, img_list, name_list, gt_list, out_root, nclass, crop_size, target_size, stride_ratio, mean, batch_size, show=0, save_seg=0, save_img=0, save_prob=0, save_saliency=0, use_hyper=0, hyper_downsample_rate=1, hyper_centroids_name=None, score_name='score', start=0, end=-1, gpu=-1, LUT=None, f=None):
    # preparing data
    if save_seg:
        if not os.path.isdir(out_root + '/seg/'):
            os.makedirs(out_root + '/seg/')
        if not os.path.isdir(out_root + '/vlz/'):
            os.makedirs(out_root + '/vlz/')
    if save_prob:
        raise NotImplementedError("save prob not implemented")

    if end==-1:
        img_list = img_list[start:]
    elif end <= len(img_list):
        img_list = img_list[start:end]
    else:
        raise Exception('end should not be larger than img_list length')

    base_size = [crop_size, crop_size]
    # grid
    hist = np.zeros((nclass, nclass))
    img0 = cv2.imread(img_list[0])
    stride = int(np.ceil(crop_size * stride_ratio))
    hgrid_num = int(np.ceil((img0.shape[0]-crop_size) / float(stride))) + 1
    assert (hgrid_num - 1) * stride + crop_size - img0.shape[0] < crop_size * 0.05
    wgrid_num = int(np.ceil((img0.shape[1]-crop_size) / float(stride))) + 1
    assert (wgrid_num - 1) * stride + crop_size - img0.shape[1] < crop_size * 0.05
    mean_map = np.tile(mean, [base_size[0], base_size[1], 1]) # H x W x C
    # caffe model init
    caffe_tester = tester.Tester(net, model, gpu)
    if use_hyper:
        if hyper_centroids_name in caffe_tester.blobs.keys():
            hyper_dsr = hyper_downsample_rate
            hyper_total = (crop_size/hyper_dsr) * (crop_size/hyper_dsr)
            hyper_yc = np.tile(np.arange(0, crop_size, hyper_dsr).reshape((crop_size/hyper_dsr, 1)), [1, crop_size/hyper_dsr]).reshape((hyper_total, 1))
            hyper_xc = np.tile(np.arange(0, crop_size, hyper_dsr).reshape((1, crop_size/hyper_dsr)), [crop_size/hyper_dsr, 1]).reshape((hyper_total, 1))
            hyper_centroids_blob = np.tile(np.hstack((hyper_yc, hyper_xc)), [batch_size, 1, 1])
            caffe_tester.blobs[hyper_centroids_name].data.flat = hyper_centroids_blob.flat
        else:
            raise Exception("Can not find the blob: %s" % hyper_centroids_name)

    # loop list
    for i in range(0, len(img_list)):
        if i%10 == 0:
            print('Processing: %d/%d' % (i, len(img_list)))
        if not os.path.isfile(img_list[i]):
            raise Exception('file not exist: %s' % (img_list[i]))
            sys.exit(1)
        img = cv2.imread(img_list[i]) # BGR, 0-255
        H,W = img.shape[:2]
        hstart = stride * np.arange(hgrid_num)
        wstart = stride * np.arange(wgrid_num)
        batch_data = np.array([(crop_images.crop_padding(img, (wst, hst, crop_size, crop_size), mean) - mean_map).astype(np.float32).transpose(2,0,1) for hst in hstart for wst in wstart])

        assert batch_data.shape[0] >= batch_size
        assert batch_data.shape[0] % batch_size == 0

        #
        if gt_list is not None:
            gt_img = cv2.imread(gt_list[i])[:,:,0]
        ensemble_prob = np.zeros((nclass, H, W), dtype=np.float32)
        ensemble_cls = np.zeros((H, W), dtype=np.uint8)
        # loop crops by batch
        for j in range(batch_data.shape[0] / batch_size):
            inputs = {'data': batch_data[j*batch_size:(j+1)*batch_size,...]}
            qblobs = caffe_tester.predict(inputs, {}, [score_name])
            if use_hyper and len(qblobs[score_name].shape) != 2:
                raise Exception("for hypercolumn, qblobs[score_name] should have 2 axis")
            for k in range(0, batch_size):
                if use_hyper:
                    prob_map = qblobs[score_name].reshape((batch_size, crop_size/hyper_dsr, crop_size/hyper_dsr,-1))[k,...].transpose((2,0,1))[:nclass,...]
                else:
                    prob_map = qblobs[score_name][k,nclass,...]
                if prob_map.max() > 1 or prob_map.min() < 0:
                    raise Exception("should with softmax")
                prob_map = np.array([cv2.resize(pm, (crop_size, crop_size)) for pm in prob_map])
                hid, wid = (j * batch_size + k) // wgrid_num, (j * batch_size + k) % wgrid_num
                ensemble_prob[:, hid*stride:hid*stride+crop_size, wid*stride:wid*stride+crop_size] += prob_map # accumulate probability

        ensemble_cls = ensemble_prob.argmax(axis=0)
        if target_size is not None:
            ensemble_cls = cv2.resize(ensemble_cls, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        ensemble_vlz = np.uint8(LUT[ensemble_cls])
        # mIU
        if gt_list is not None:
            ensemble_cls[ensemble_cls >= nclass] = 0
            hist += fast_hist(gt_img.flatten(), ensemble_cls.flatten(), nclass)
        if show:
            cv2.imshow("image", img)
            cv2.imshow("seg result", ensemble_vlz)
            cv2.waitKey(0)
        if save_seg:
            cls_map_fn = out_root + '/seg/' + name_list[i] + ".png"
            out_map_fn = out_root + "/vlz/" + name_list[i] + ".png"
            if not os.path.isdir(os.path.dirname(cls_map_fn)):
                os.makedirs(os.path.dirname(cls_map_fn))
            if not os.path.isdir(os.path.dirname(out_map_fn)):
                os.makedirs(os.path.dirname(out_map_fn))
            cv2.imwrite(cls_map_fn, ensemble_cls)
            cv2.imwrite(out_map_fn, ensemble_vlz)
    if gt_list is not None:
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
    parser.add_argument('--crop_size', type=int)
    parser.add_argument('--target_h', type=int, default=None)
    parser.add_argument('--target_w', type=int, default=None)
    parser.add_argument('--stride_ratio', type=float)
    parser.add_argument('--mean', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--gt_root', type=str, default=None)
    parser.add_argument('--out_root', type=str, default=None)
    parser.add_argument('--show', type=int, default=0)
    parser.add_argument('--save_seg', type=int, default=0)
    parser.add_argument('--save_img', type=int, default=0)
    parser.add_argument('--save_prob', type=int, default=0)
    parser.add_argument('--save_saliency', type=int, default=0)
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
        #mean = np.array([104.,117.,123.])
    elif args.mean == 'zero_mean':
        mean = np.array([0.0, 0.0, 0.0])
    else:
        print('Unknown mean value type: %s' % args.mean)
        sys.exit(1)
    if args.output_fn is not None:
        f = open(args.output_fn, 'a')
    elif args.out_root is not None and args.gt_root is not None:
        f = open(args.out_root + "/results.txt", 'a')
    else:
        f = None
    nclass = 19
    LUT = np.load(os.path.dirname(__file__) + '/CityScapes_color_LUT_19.npy')[:,::-1]
    with open(args.test_list, 'r') as infile:
        lines = [line for line in infile.readlines()]
        img_list = [args.test_root + line.strip().split(' ')[0] for line in lines]
        name_list = [fn.split('/')[-2]+'/'+fn.split('/')[-1][:-16] for fn in img_list]
        if args.gt_root is not None:
            gt_list = [args.gt_root + line.strip().split(' ')[1] for line in lines]
        else:
            gt_list = None
    if args.target_h is not None and args.target_w is not None:
        target_size = [args.target_h, args.target_w]
    else:
        target_size = None
    test_eval_seg(args.net, args.model, img_list, name_list, gt_list, args.out_root, nclass, args.crop_size, target_size, args.stride_ratio, mean, args.batch_size, args.show, args.save_seg, args.save_img, args.save_prob, args.save_saliency, args.use_hyper, args.hyper_downsample_rate, args.hyper_centroids_name, args.score_name, args.start, args.end, args.gpu_id, LUT, f)
    if f is not None:
        f.close()

if __name__ == "__main__":
    main_eval()
