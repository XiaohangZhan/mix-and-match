import cv2
import numpy as np
import os

def resize_pad_to_fit(img, size, pad_value=np.array([0,0,0]), interp=cv2.INTER_LINEAR): # size: [rows, cols], to uniform size
    if len(size) != 2:
        raise Exception("Error: len(size) should be 2")   
    rows = img.shape[0]
    cols = img.shape[1]
    two_axis = False
    if len(img.shape) == 2:
        two_axis = True
        img = img[:,:,np.newaxis]
        assert len(pad_value) == 1
    else:
        assert len(pad_value) == 3
    channels = img.shape[2]
    pad_map = np.tile(pad_value, [size[0], size[1], 1])
    out_img = pad_map.copy().astype(img.dtype)
    aspect_img = float(rows) / cols
    aspect_new = float(size[0]) / size[1]
    if aspect_img > aspect_new:
        new_cols = int(size[0] / aspect_img)
        resize_img = [cv2.resize(img[:,:,c], (new_cols, size[0]), interpolation=interp) for c in range(channels)]
        for c in range(channels):
            out_img[:, (size[1] - new_cols) / 2 : (size[1] - new_cols) / 2 + new_cols, c] = resize_img[c]
    else:
        new_rows = int(size[1] * aspect_img)
        resize_img = [cv2.resize(img[:,:,c], (size[1], new_rows), interpolation=interp) for c in range(channels)]
        for c in range(channels):
            out_img[(size[0] - new_rows) / 2 : (size[0] - new_rows) / 2 + new_rows, :, c] = resize_img[c]
    if two_axis:
        out_img = out_img[:,:,0]
    return out_img

def resize_crop_to_fit(img, size, interp=cv2.INTER_LINEAR):
    '''
    size: [rows, cols], list or tuple
    '''
    if len(size) != 2:
        raise Exception("Error: len(size) should be 2")
    rows = img.shape[0]
    cols = img.shape[1]
    two_axis = False
    if len(img.shape) == 2:
        two_axis = True
        img = img[:,:,np.newaxis]
    channels = img.shape[2]
    aspect_img = float(rows) / cols
    aspect_new = float(size[0]) / size[1]
    if aspect_img > aspect_new:
        crop_rows = int(cols * aspect_new)
        crop_img = img[(rows - crop_rows) / 2: (rows - crop_rows) / 2 + crop_rows, :]
        out_img = np.concatenate([cv2.resize(crop_img[:,:,c], (size[1], size[0]), interpolation=interp)[:,:,np.newaxis] for c in range(channels)], axis=2)
    else:
        crop_cols = int(rows / aspect_new)
        crop_img = img[:, (cols - crop_cols) / 2 : (cols - crop_cols) / 2 + crop_cols, :]
        out_img = np.concatenate([cv2.resize(crop_img[:,:,c], (size[1], size[0]), interpolation=interp)[:,:,np.newaxis] for c in range(channels)], axis=2)
    if two_axis:
        out_img = out_img[:,:,0]
    return out_img


def resize_crop_list(root_dir, list_filename, backend, out_dir, size):
    with open(list_filename, 'r') as infile:
        for line in infile:
            line = line.rstrip('\n').split(' ')[0]
            print line
            if len(line) > 0:
                img = cv2.imread(root_dir + line + backend)
                img_uniform = resize_pad_to_fit(img, size)
                out_fn = out_dir + line + backend
                if not os.path.isdir(out_fn[:out_fn.rfind('/')]):
                    os.makedirs(out_fn[:out_fn.rfind('/')])
                cv2.imwrite(out_fn, img_uniform)
def resize_crop_list_multithread(root_dir, list_filename, backend, out_dir, size, start_point, nthread, pad_value, interp):
    import threading
    with open(list_filename, 'r') as infile:
        lines = infile.readlines()
    lines = lines[start_point:]
    total = len(lines)
    print "total: %d" % total
    print "num thread: %d" % nthread
    block = np.ceil(float(total) / nthread)
    start_list = [i * block for i in range(nthread)]
    end_list = [(i + 1) * block for i in range(nthread)]
    end_list[-1] = total
    def sub_thread(tid, start, end):
        for idx, line in enumerate(lines[start:end]):
            if idx % 1000 == 0:
                print('sub thread %d: %d/%d' % (tid, idx, end-start))
            line = line.rstrip('\n').split(' ')[0]
            if len(line) > 0:
                img = cv2.imread(root_dir + line + backend)
                img_uniform = resize_pad_to_fit(img, size, pad_value, interp)
                out_fn = out_dir + line + backend
                if not os.path.isdir(out_fn[:out_fn.rfind('/')]):
                    os.makedirs(out_fn[:out_fn.rfind('/')])
                cv2.imwrite(out_fn, img_uniform)
    for t in range(nthread):
        th = threading.Thread(target = sub_thread, args = (t, int(start_list[t]), int(end_list[t])))
        th.start()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 3:
        root_dir = sys.argv[1]
        list_filename = sys.argv[2]
        backend = ".png"
        out_dir = sys.argv[3]
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        size = [480, 480]
        start_point = 0
        #resize_crop_list(root_dir, list_filename, backend, out_dir, size)
        resize_crop_list_multithread(root_dir, list_filename, backend, out_dir, size, start_point, nthread=10, pad_value=255, interp=cv2.INTER_NEAREST)
