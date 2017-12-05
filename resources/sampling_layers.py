import os
import warnings
import numpy as np
import cv2
import create_patch
import caffe
import pdb

def get_patch(bottom_data, c, output_size):
    h, w = bottom_data.shape[2:]
    patch_img = np.zeros((3, c[4], c[4]))
    im_r0 = max(0, c[2]-c[4]/2)
    im_c0 = max(0, c[3]-c[4]/2)
    im_r1 = min(h, c[2]+c[4]/2)
    im_c1 = min(h, c[3]+c[4]/2)
    p_r0 = max(0, c[4]/2-c[2])
    p_c0 = max(0, c[4]/2-c[3])
    p_r1 = min(c[4], h+c[4]/2-c[2])
    p_c1 = min(c[4], w+c[4]/2-c[3])
    patch_img[:, p_r0:p_r1, p_c0:p_c1] = bottom_data[c[1], :, im_r0:im_r1, im_c0:im_c1].copy()
    patch_img = patch_img.transpose((1,2,0))
    return cv2.resize(patch_img, (output_size, output_size)).transpose((2,0,1)).astype(np.float)

class RandomSamplingLayer(caffe.Layer):
    def setup(self, bottom, top):
        warnings.filterwarnings("ignore")
        params = eval(self.param_str)
        self.output_size = params['output_size']
        self.num = params['num']
        self.by_ovlp = params['by_ovlp']
        self.minsz = params['minsz']
        self.maxsz = params['maxsz']
        if self.num % bottom[0].data.shape[0] != 0:
            raise Exception("num should be divided by batch size.")
        self.num_cand = self.num / bottom[0].data.shape[0]
        if len(top) != 2:
            raise Exception("Need exact two tops.")
        if len(bottom) != 2:
            raise Exception("Need exact two bottoms.")
    def reshape(self, bottom, top):
        top[0].reshape(self.num, bottom[0].data.shape[1], self.output_size, self.output_size)
        top[1].reshape(self.num, 1)
    def forward(self, bottom, top):
        idx = 0
        for i in range(bottom[0].data.shape[0]):
            img = bottom[0].data[i,...].transpose((1,2,0)).copy()
            seg = bottom[1].data[i,...].transpose((1,2,0)).copy()
            patch, cls = create_patch.createRandomPatchImg(img, seg, self.num_cand, [self.minsz, self.maxsz], 0.1, self.output_size, by_ovlp=self.by_ovlp, show=False)
            if patch.shape[0] != self.num_cand:
                raise Exception("Number of patches not consistent: %d vs. %d" % (patch.shape[0], self.num_cand))
            for k in range(patch.shape[0]):
                top[0].data[idx,...] = get_patch(bottom[0].data, np.hstack((np.array([cls[k], i]), patch[k,:])), self.output_size)
                top[1].data[idx,:] = cls[k]
                idx += 1
        # check
        if False:
            if not os.path.isdir('output/class/'):
                os.makedirs('output/class/')
            show_data = top[0].data.copy()
            show_data[:,0,...] += 104
            show_data[:,1,...] += 117
            show_data[:,2,...] += 123
            num = np.zeros((21,), dtype=np.int)
            for i in range(show_data.shape[0]):
                cv2.imwrite('output/class/' + str(top[1].data[i,:].astype(np.int)) + '_' + str(num[top[1].data[i,:].astype(np.int)])+ '.jpg', show_data[i,...].transpose((1,2,0)).astype(np.uint8))
                num[top[1].data[i,:].astype(np.int)] += 1
            pdb.set_trace()
    def backward(self, top, propagate_down, bottom):
        pass

class GraphToTripletLayer(caffe.Layer):
    def setup(self, bottom, top):
        warnings.filterwarnings("ignore")
        self.N = bottom[0].data.shape[0]
        if len(top) != 3:
            raise Exception("Need exact three tops")
        if len(bottom) != 2:
            raise Exception("Need exact two bottoms")
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)
        top[1].reshape(*bottom[0].data.shape)
        top[2].reshape(*bottom[0].data.shape)
    def forward(self, bottom, top):
        in_graph = []
        self.triplet_idx = -1 * np.ones((self.N, 3), dtype=np.int)
        labels = bottom[1].data[...]
        for i in np.random.permutation(self.N):
            self.triplet_idx[i,0] = i
            label = labels[i]
            pos_cand = [idx for idx in in_graph if labels[idx] == label]
            neg_cand = [idx for idx in in_graph if labels[idx] != label]
            if len(pos_cand) != 0:
                self.triplet_idx[i,1] = np.random.choice(pos_cand)
            if len(neg_cand) != 0:
                self.triplet_idx[i,2] = np.random.choice(neg_cand)
            in_graph.append(i)
        for i in range(self.N):
            if self.triplet_idx[i,1] == -1:
                label = labels[i]
                pos_cand = [idx for idx in in_graph if labels[idx] == label and idx != i]
                if len(pos_cand) != 0:
                    self.triplet_idx[i,1] = np.random.choice(pos_cand)
                else:
                    self.triplet_idx[i,1] = i
            if self.triplet_idx[i,2] == -1:
                label = labels[i]
                neg_cand = [idx for idx in in_graph if labels[idx] != label]
                if len(neg_cand) != 0:
                    self.triplet_idx[i,2] = np.random.choice(neg_cand)
        for i in range(self.N):
            top[0].data[i,...] = bottom[0].data[self.triplet_idx[i,0],...]
            top[1].data[i,...] = bottom[0].data[self.triplet_idx[i,1],...]
            if self.triplet_idx[i,2] != -1:
                top[2].data[i,...] = bottom[0].data[self.triplet_idx[i,2],...]
            else:
                top[2].data[i,...] = 0.
    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = 0.
        for i in range(self.N):
            bottom[0].diff[self.triplet_idx[i,0],...] += top[0].diff[i,...]
            bottom[0].diff[self.triplet_idx[i,1],...] += top[1].diff[i,...]
            if self.triplet_idx[i,2] != -1:
                bottom[0].diff[self.triplet_idx[i,2],...] += top[2].diff[i,...]
