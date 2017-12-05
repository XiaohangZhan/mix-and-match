import cv2
import numpy as np
import random
#import resize_uniform
#import vlz_seg_color
import time
import pdb
size_list = [0.6, 0.5, 0.4, 0.3, 0.2]

def inSize(center,size, h, w):
    return (center[0] - size/2 >= 0 and center[0] + size/2 < h and center[1] - size/2 >= 0 and center[1] + size/2 < w)
def computeOverlapGeneral(seg, center, size, cls):
    h,w = seg.shape[:2]
    r0 = max(0, int(center[0]-size/2))
    c0 = max(0, int(center[1]-size/2))
    r1 = min(h, int(center[0]+size/2))
    c1 = min(w, int(center[1]+size/2))
    return (seg[r0:r1, c0:c1, 0] == cls).sum() / float(size*size)
def computeOverlap(seg, center, size, cls):
    if not inSize(center, size, seg.shape[0], seg.shape[1]):
        raise Exception('Query patch out of Seg range.')
    return (seg[int(center[0]-size/2):int(center[0]+size/2), int(center[1]-size/2):int(center[1]+size/2), 0]==cls).sum() / float(size*size)
def computeBBoxIoU(center0, size0, center1, size1):
    lx = max(center0[1]-size0/2, center1[1]-size1/2)
    rx = min(center0[1]+size0/2, center1[1]+size1/2)
    uy = max(center0[0]-size0/2, center1[0]-size1/2)
    dy = min(center0[0]+size0/2, center1[0]+size1/2)
    interArea = (rx-lx)*(dy-uy)
    return interArea / float(size0*size0 + size1*size1 - interArea)
def computeBBoxOvlpRate(center0, size0, center1, size1):
    lx = max(center0[1]-size0/2, center1[1]-size1/2)
    rx = min(center0[1]+size0/2, center1[1]+size1/2)
    uy = max(center0[0]-size0/2, center1[0]-size1/2)
    dy = min(center0[0]+size0/2, center1[0]+size1/2)
    interArea = (rx-lx)*(dy-uy)
    smallArea = min(size0*size0, size1*size1)
    return interArea / float(smallArea)
def findMaxOverlap(seg, center, size, cls_list):
    overlaps = np.array([computeOverlap(seg, center, size, c) for c in cls_list], dtype=np.float)
    return cls_list[overlaps.argmax()], overlaps.max()

def patchDeDup(patches, prior, h, w):
    mask = np.zeros((h,w))
    first_area = float(patches[0,2]*patches[0,2])
    keep = []
    for i in prior:
        p = patches[i,:]
        sub_area = mask[max(0,p[0]-p[2]/2) : min(p[0]+p[2]/2,h), max(0,p[1]-p[2]/2) : min(p[1]+p[2]/2, w)].sum()
        if sub_area / first_area > 0.6:
            continue
        mask[max(0,p[0]-p[2]/2) : min(p[0]+p[2]/2,h), max(0,p[1]-p[2]/2) : min(p[1]+p[2]/2, w)] = 1
        keep.append(i)
    return keep

def createRandomPatchImg(img, seg, num, sz_range, bg_ratio, output_size, by_ovlp=False, show=1): # img: CxHxW, seg: 1xHxW, query_size: better to be even
    start_time = time.time()
    w = img.shape[1]
    h = img.shape[0]
    bg_num = max(1, int(num*bg_ratio))
    fg_num = max(1, num - bg_num)
    seg = seg.astype(int)
    if len(seg.shape) != 3: 
        raise Exception('Seg should contain 3 dims.')
    if (seg.shape[0] != h or seg.shape[1] != w):
        raise Exception('img and seg size not consistent.')
    fg_scope = np.where((seg[:,:,0] != 0) & (seg[:,:,0] != 255))
    bg_scope = np.where(seg[:,:,0] == 0)
    if len(bg_scope[0]) == 0 and len(fg_scope[0]) == 0:
        bg_scope = np.where((seg[:,:,0] == 0) | (seg[:,:,0] == 255))
    if len(fg_scope[0]) == 0:
        fg_scope = bg_scope
    if len(bg_scope[0]) == 0:
        bg_scope = fg_scope
    fg_seed = np.random.randint(0, len(fg_scope[0]), size=5*fg_num)
    bg_seed = np.random.randint(0, len(bg_scope[0]), size=5*bg_num)
    fg_patch = np.array([(fg_scope[0][r], fg_scope[1][r], np.random.randint(sz_range[0], sz_range[1]+1)/2*2) for r in fg_seed]).astype(np.int)
    bg_patch = np.array([(bg_scope[0][r], bg_scope[1][r], np.random.randint(sz_range[0], sz_range[1]+1)/2*2) for r in bg_seed]).astype(np.int)
    fg_cls = np.array([seg[fg_scope[0][r], fg_scope[1][r], 0] for r in fg_seed]).astype(np.int)
    bg_cls = np.array([0 for r in bg_seed]).astype(np.int)
    if by_ovlp:
        fg_ovlp = np.array([computeOverlapGeneral(seg, [p[0], p[1]], p[2], c) for p,c in zip(fg_patch, fg_cls)])
        bg_ovlp = np.array([computeOverlapGeneral(seg, [p[0], p[1]], p[2], c) for p,c in zip(bg_patch, bg_cls)])
        fg_prior = np.argsort(fg_ovlp)[::-1]
        bg_keep = np.argsort(bg_ovlp)[::-1][:bg_num]
    else:
        fg_prior = np.arange(fg_patch.shape[0])
        bg_keep = np.arange(bg_num)

    fg_keep = patchDeDup(fg_patch, fg_prior, h, w)
    if len(fg_keep) < fg_num:
        candi = [ci for ci in fg_prior if ci not in fg_keep]
        fg_keep.extend(candi[:fg_num-len(fg_keep)])
    elif len(fg_keep) > fg_num:
        fg_keep = fg_keep[:fg_num]
    fg_keep = np.array(fg_keep).astype(np.int)
    fg_patch = fg_patch[fg_keep, :]
    bg_patch = bg_patch[bg_keep, :]
    fg_cls = fg_cls[fg_keep]
    bg_cls = bg_cls[bg_keep]
    if by_ovlp:
        fg_ovlp = fg_ovlp[fg_keep]
        bg_ovlp = bg_ovlp[bg_keep]
        ovlp = np.hstack((fg_ovlp, bg_ovlp))
    patch = np.vstack((fg_patch, bg_patch))
    cls = np.hstack((fg_cls, bg_cls))
    if show:
        show_img = img.copy()
        if show_img.min() < 0:
            show_img[:,:,0] += 104
            show_img[:,:,1] += 117
            show_img[:,:,2] += 123
        show_img = show_img.astype(np.uint8)
        #print query_center
        for i,p in enumerate(patch):
            cv2.rectangle(show_img, (p[1]-p[2]/2,p[0]-p[2]/2), (p[1]+p[2]/2,p[0]+p[2]/2), (255,0,0))
            if by_ovlp:
                cv2.putText(show_img, str(cls[i]) + " " + str(ovlp[i]), (p[1]-p[2]/2, p[0]-p[2]/2+12), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=1)
            else:
                cv2.putText(show_img, str(cls[i]), (p[1]-p[2]/2, p[0]-p[2]/2+12), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=1)
            cv2.imshow('Showing Patches', show_img)
            cv2.waitKey(0)
    return patch, cls

def createSinglePatchImg(img, seg, query, num, output_size, overlap_th, pos_bbox_iou, show=1): # img: CxHxW, seg: 1xHxW, query_size: better to be even
    '''
    INPUTS:
    img: HxWxC
    seg: HxWx1 or HxWx3
    npos: number of positive patches
    nneg: number of negative patches
    query_size: int, size of query patch
    query_ignore: query ignoring list
    output_size: int, size of output patches after resize
    overlap_th1: max overlap with positive for selecting negatives
    overlap_th2: min overlap with positive for selecting positives
    mode: 0 ---- patch class = max overlapped class
          1 ---- patch class = central pixel class
    '''
    start_time = time.time()
    w = img.shape[1]
    h = img.shape[0]
    seg = seg.astype(int)
    if len(seg.shape) != 3: 
        raise Exception('Seg should contain 3 dims.')
    if (seg.shape[0] != h or seg.shape[1] != w):
        raise Exception('img and seg size not consistent.')
    ## find query patch
    find_query_patch = 0
    query_patch = -1*np.ones((num, 3), dtype=np.int)
    max_idx = query
    query_scope = np.where(seg[:,:,0] == max_idx)
    ratio = np.sqrt(float(len(query_scope[0])) / (h*w))
    if query == 0:
        query_size_list = [0.5, 0.4, 0.3, 0.2]
    else:
        query_size_list = np.linspace(ratio, max(size_list[-1], ratio/2), 5)
    while (find_query_patch < num):
        if len(query_scope[0]) == 0:
            break
        find_curr = False
        for rand_id in range(1000):
            randseed = np.random.randint(0,len(query_scope[0]))
            query_center = [query_scope[0][randseed], query_scope[1][randseed]]
            for s in query_size_list:
                if not inSize(query_center, s*h, h, w):
                    continue
                if np.any(np.array([computeBBoxOvlpRate(query_center, s*h, query_patch[e,0:2], query_patch[e,2]) for e in range(find_query_patch)]) > pos_bbox_iou):
                    continue
                ovlp = computeOverlap(seg, query_center, s*h, max_idx)
                if ovlp > overlap_th:  
                    find_curr = True
                    query_patch[find_query_patch, :] = np.array(query_center + [int(s*float(h))])
                    find_query_patch += 1   
                    break
            if find_curr:
                break
        if not find_curr:
            break
    #print("execution time: %s seconds" % (time.time() - start_time))
    # visualize
    if show:
        show_img = img.copy()
        #print query_center
        for i in range(find_query_patch):
            cv2.rectangle(show_img, (query_patch[i,1]-query_patch[i,2]/2,query_patch[i,0]-query_patch[i,2]/2), (query_patch[i,1]+query_patch[i,2]/2,query_patch[i,0]+query_patch[i,2]/2), (255,0,0))
        #seg_color = vlz_seg_color.convert_label_color_map(seg).copy()
        cv2.imshow('Showing Patches', show_img)
        #cv2.imshow('Input seg', seg_color)
        cv2.waitKey(0)
    return find_query_patch, query_patch

def createTripletPatchImg(img, seg, query_max_portion, query_ignore, output_size, overlap_th1, overlap_th2, pos_bbox_iou,  mode = 0, show=0): # img: CxHxW, seg: 1xHxW, query_size: better to be even
    '''
    INPUTS:
    img: HxWxC
    seg: HxWx1 or HxWx3
    npos: number of positive patches
    nneg: number of negative patches
    query_size: int, size of query patch
    query_ignore: query ignoring list
    output_size: int, size of output patches after resize
    overlap_th1: max overlap with positive for selecting negatives
    overlap_th2: min overlap with positive for selecting positives
    mode: 0 ---- patch class = max overlapped class
          1 ---- patch class = central pixel class
    '''
    start_time = time.time()
    w = img.shape[1]
    h = img.shape[0]
    seg = seg.astype(int)
    if len(seg.shape) != 3: 
        raise Exception('Seg should contain 3 dims.')
    if (seg.shape[0] != h or seg.shape[1] != w):
        raise Exception('img and seg size not consistent.')
    if not (mode == 0 or mode == 1):
        raise Exception('Unknown mode: %d' % mode)
    if (query_max_portion and mode == 0):
        raise Exception('Not implemented for (query_max_portion and mode == 0)')
    ## find query patch
    find_query_patch = False
    if query_max_portion:
        query_cls = 0
        hist = np.array([(seg[:,:,0]==i).sum() for i in range(21) if i not in query_ignore])
        index = [i for i in range(21) if i not in query_ignore]
        max_idx = index[np.argmax(hist)]
        ratio = np.sqrt(float(hist.max()) / (h*w))
        query_size_list = np.linspace(ratio, max(size_list[-1], ratio/2), 5)
        query_scope = np.where(seg[:,:,0] == max_idx)
        for rand_id in range(2000):
            if len(query_scope[0]) == 0:
                break
            randseed = np.random.randint(0,len(query_scope[0]))
            query_center = [query_scope[0][randseed], query_scope[1][randseed]]
            for s in query_size_list:
                if not inSize(query_center, s*h, h, w):
                    continue
                ovlp = computeOverlap(seg, query_center, s*h, max_idx)
                if ovlp > overlap_th2:  
                    find_query_patch = True
                    query_patch = np.array(query_center + [int(s*float(h))])
                    query_cls = max_idx
                    break
            if find_query_patch:
                break
    else:
        query_cls = 0
        for rand_id in range(2000):
            query_center = [random.randint(0, h), random.randint(0, w)]
            for s in size_list:
                if not inSize(query_center, s*h, h, w):
                    continue
                if mode == 0:
                    max_idx,max_ovlp = findMaxOverlap(seg, query_center, s*h, [i for i in range(21) if i not in query_ignore])
                else:
                    max_idx = seg[query_center[0], query_center[1], 0]
                    if max_idx not in [i for i in range(21) if i not in query_ignore]:
                        continue
                    max_ovlp = computeOverlap(seg, query_center, s*h, max_idx)
                if max_ovlp > overlap_th2:
                    find_query_patch = True
                    query_patch = np.array(query_center + [int(s*float(h))])
                    query_cls = max_idx
                    break
            if find_query_patch:
                break
    ## find positive patches
    total_iter = 0
    find_pos = False
    if mode == 1:
        if query_max_portion:
            pos_scope = query_scope
            pos_size_list = query_size_list
        else:
            pos_scope = np.where(seg[:,:,0] == query_cls)
            pos_size_list = size_list
        neg_scope = np.where((seg[:,:,0] != query_cls) & (seg[:,:,0] != 255))
    while (not find_pos):
        if total_iter > 2000:
            break
        total_iter += 1
        if mode == 0:
            sample_center = [random.randint(0, h), random.randint(0, w)]
        else:
            if len(pos_scope[0]) == 0:
                break
            randseed = np.random.randint(0,len(pos_scope[0]))
            sample_center = [pos_scope[0][randseed], pos_scope[1][randseed]]
        for s in pos_size_list:
            if not inSize(sample_center, s*h, h, w):
                continue
            if find_query_patch:
                bbox_iou = computeBBoxIoU(sample_center, s*h, query_patch[:2], query_patch[2])
                if bbox_iou > pos_bbox_iou:
                    continue
            ovlp = computeOverlap(seg, sample_center, s*h, query_cls)
            if (ovlp > overlap_th2):
                find_pos = True
                pos_patch = np.array(sample_center + [int(s*float(h))])
                break
    ## find negative patches
    total_iter = 0
    find_neg = False
    while (not find_neg):
        if total_iter > 2000:
            break
        total_iter += 1
        if mode == 0:
            sample_center = [random.randint(0, h), random.randint(0, w)]
        else:
            if len(neg_scope[0]) == 0:
                break
            randseed = np.random.randint(0,len(neg_scope[0]))
            sample_center = [neg_scope[0][randseed], neg_scope[1][randseed]]
        for s in size_list:
            if not inSize(sample_center, s*h, h, w):
                continue
            ovlp = computeOverlap(seg, sample_center, s*h, query_cls)
            if (ovlp < overlap_th1 and computeOverlap(seg, sample_center, s*h, 255) < overlap_th1):
                find_neg = True
                neg_patch = np.array(sample_center + [int(s*float(h))])
                break

    # output blobs
    output_blob = np.zeros((3,3,output_size, output_size), dtype=np.float)
    if find_query_patch:
        query_crop = img[query_patch[0]-query_patch[2]/2:query_patch[0]+query_patch[2]/2, query_patch[1]-query_patch[2]/2:query_patch[1]+query_patch[2]/2,:].copy()
        output_blob[0,...] = cv2.resize(query_crop, (output_size, output_size)).transpose((2,0,1)).astype(np.float)
    if find_pos:
        pos_crop = img[pos_patch[0]-pos_patch[2]/2:pos_patch[0]+pos_patch[2]/2, pos_patch[1]-pos_patch[2]/2:pos_patch[1]+pos_patch[2]/2, :].copy()
        output_blob[1,...] = cv2.resize(pos_crop, (output_size, output_size)).transpose((2,0,1)).astype(np.float)
    else:
        output_blob[1,...] = output_blob[0,...]
    if find_neg:
        neg_crop = img[neg_patch[0]-neg_patch[2]/2:neg_patch[0]+neg_patch[2]/2, neg_patch[1]-neg_patch[2]/2:neg_patch[1]+neg_patch[2]/2, :].copy()
        output_blob[2,...] = cv2.resize(neg_crop, (output_size, output_size)).transpose((2,0,1)).astype(np.float)
    #print("execution time: %s seconds" % (time.time() - start_time))
    # visualize
    if show:
        COLOR = [(0,0,255),(0,255,0)]
        show_img = img.copy()
        #print query_center
        if find_query_patch:
            cv2.rectangle(show_img, (query_patch[1]-query_patch[2]/2,query_patch[0]-query_patch[2]/2), (query_patch[1]+query_patch[2]/2,query_patch[0]+query_patch[2]/2), (255,0,0))
        if find_pos:
            cv2.rectangle(show_img, (pos_patch[1]-pos_patch[2]/2,pos_patch[0]-pos_patch[2]/2), (pos_patch[1]+pos_patch[2]/2,pos_patch[0]+pos_patch[2]/2), COLOR[1])
        if find_neg:
            cv2.rectangle(show_img, (neg_patch[1]-neg_patch[2]/2,neg_patch[0]-neg_patch[2]/2), (neg_patch[1]+neg_patch[2]/2,neg_patch[0]+neg_patch[2]/2), COLOR[0])
        seg_color = vlz_seg_color.convert_label_color_map(seg).copy()
        cv2.imshow('Showing Patches', show_img)
        cv2.imshow('Input seg', seg_color)
        cv2.waitKey(0)
    return output_blob
    #return output_data_blob, output_cls_blob
def createTripletPatchTwoImg(img0, img1, seg0, seg1, query_max_portion, fix_query, query_ignore, output_size, overlap_th1, overlap_th2, mode = 0, show=0): # img: CxHxW, seg: 1xHxW, query_size: better to be even
    '''
    INPUTS:
    img: HxWxC
    seg: HxWx1 or HxWx3
    npos: number of positive patches
    nneg: number of negative patches
    query_size: int, size of query patch
    query_ignore: query ignoring list
    output_size: int, size of output patches after resize
    overlap_th1: max overlap with positive for selecting negatives
    overlap_th2: min overlap with positive for selecting positives
    fix_query: -1 ---- not fix
               0-20 ---- fix to specific class
    mode: 0 ---- patch class = max overlapped class
          1 ---- patch class = central pixel class
    '''
    start_time = time.time()
    w = img0.shape[1]
    h = img0.shape[0]
    seg0 = seg0.astype(int)
    seg1 = seg1.astype(int)
    if len(seg0.shape) != 3 or len(seg1.shape) != 3: 
        raise Exception('Seg should contain 3 dims.')
    if (seg0.shape[0] != h or seg0.shape[1] != w or seg1.shape[0] != h or seg1.shape[1] != w or img1.shape[0] != h or img1.shape[1] != w):
        raise Exception('All imgs and segs size not consistent.')
    if not (mode == 0 or mode == 1):
        raise Exception('Unknown mode: %d' % mode)
    if (query_max_portion and mode == 0):
        raise Exception('Not implemented for (query_max_portion and mode == 0)')
    ## find query patch
    find_query_patch = False
    if query_max_portion:
        query_cls = 0
        if (fix_query >=0 and fix_query < 21):
            max_idx = fix_query
            ratio = np.sqrt(float((seg[:,:,0]==max_idx).sum()) / (h*w))
        elif fix_query == -1:
            count = [(seg[:,:,0]==i).sum() for i in range(1,21)]
            index = [i for i in range(21) if i not in query_ignore]
            hist = np.array([count[i] for i in index])
            max_idx = index[np.argmax(hist)]
            ratio = np.sqrt(float(hist.max()) / (h*w))
        query_size_list = np.linspace(ratio, max(size_list[-1], ratio/2), 5)
        query_scope = np.where(seg0[:,:,0] == max_idx)
        for rand_id in range(2000):
            if len(query_scope[0]) == 0:
                break
            randseed = np.random.randint(0,len(query_scope[0]))
            query_center = [query_scope[0][randseed], query_scope[1][randseed]]
            for s in query_size_list:
                if not inSize(query_center, s*h, h, w):
                    continue
                ovlp = computeOverlap(seg0, query_center, s*h, max_idx)
                if ovlp > overlap_th2:  
                    find_query_patch = True
                    query_patch = np.array(query_center + [int(s*float(h))])
                    query_cls = max_idx
                    break
            if find_query_patch:
                break
    else:
        query_cls = 0
        for rand_id in range(2000):
            query_center = [random.randint(0, h), random.randint(0, w)]
            for s in size_list:
                if not inSize(query_center, s*h, h, w):
                    continue
                if mode == 0:
                    max_idx,max_ovlp = findMaxOverlap(seg0, query_center, s*h, [i for i in range(21) if i not in query_ignore])
                else:
                    max_idx = seg0[query_center[0], query_center[1], 0]
                    if max_idx not in [i for i in range(21) if i not in query_ignore]:
                        continue
                    max_ovlp = computeOverlap(seg0, query_center, s*h, max_idx)
                if max_ovlp > overlap_th2:
                    find_query_patch = True
                    query_patch = np.array(query_center + [int(s*float(h))])
                    query_cls = max_idx
                    break
            if find_query_patch:
                break
    ## find positive patches
    total_iter = 0
    find_pos = False
    if mode == 1:
        pos_scope = np.where(seg1[:,:,0] == query_cls)
        neg_scope = np.where((seg1[:,:,0] != query_cls) & (seg1[:,:,0] != 255))
        ratio = np.sqrt(float(len(pos_scope[0])) / (h*w))
        pos_size_list = np.linspace(ratio, max(size_list[-1], ratio/2), 5)
    else:
        pos_size_list = size_list[:]
    while (not find_pos):
        if total_iter > 2000:
            break
        total_iter += 1
        if mode == 0:
            sample_center = [random.randint(0, h), random.randint(0, w)]
        else:
            if len(pos_scope[0]) == 0:
                break
            randseed = np.random.randint(0,len(pos_scope[0]))
            sample_center = [pos_scope[0][randseed], pos_scope[1][randseed]]
        for s in pos_size_list:
            if not inSize(sample_center, s*h, h, w):
                continue
            ovlp = computeOverlap(seg1, sample_center, s*h, query_cls)
            if (ovlp > overlap_th2):
                find_pos = True
                pos_patch = np.array(sample_center + [int(s*float(h))])
                break
    ## find negative patches
    total_iter = 0
    find_neg = False
    while (not find_neg):
        if total_iter > 2000:
            break
        total_iter += 1
        if mode == 0:
            sample_center = [random.randint(0, h), random.randint(0, w)]
        else:
            if len(neg_scope[0]) == 0:
                break
            randseed = np.random.randint(0,len(neg_scope[0]))
            sample_center = [neg_scope[0][randseed], neg_scope[1][randseed]]
        for s in size_list:
            if not inSize(sample_center, s*h, h, w):
                continue
            ovlp = computeOverlap(seg1, sample_center, s*h, query_cls)
            if (ovlp < overlap_th1 and computeOverlap(seg1, sample_center, s*h, 255) < overlap_th1):
                find_neg = True
                neg_patch = np.array(sample_center + [int(s*float(h))])
                break

    # output blobs
    output_blob = np.zeros((3,3,output_size, output_size), dtype=np.float)
    if find_query_patch:
        query_crop = img0[query_patch[0]-query_patch[2]/2:query_patch[0]+query_patch[2]/2, query_patch[1]-query_patch[2]/2:query_patch[1]+query_patch[2]/2,:].copy()
        output_blob[0,...] = cv2.resize(query_crop, (output_size, output_size)).transpose((2,0,1)).astype(np.float)
    if find_pos:
        pos_crop = img1[pos_patch[0]-pos_patch[2]/2:pos_patch[0]+pos_patch[2]/2, pos_patch[1]-pos_patch[2]/2:pos_patch[1]+pos_patch[2]/2, :].copy()
        output_blob[1,...] = cv2.resize(pos_crop, (output_size, output_size)).transpose((2,0,1)).astype(np.float)
    if find_neg:
        neg_crop = img1[neg_patch[0]-neg_patch[2]/2:neg_patch[0]+neg_patch[2]/2, neg_patch[1]-neg_patch[2]/2:neg_patch[1]+neg_patch[2]/2, :].copy()
        output_blob[2,...] = cv2.resize(neg_crop, (output_size, output_size)).transpose((2,0,1)).astype(np.float)
    #print("execution time: %s seconds" % (time.time() - start_time))
    # visualize
    if show:
        COLOR = [(0,0,255),(0,255,0)]
        show_img0 = img0.copy()
        show_img1 = img1.copy()
        #print query_center
        if find_query_patch:
            cv2.rectangle(show_img0, (query_patch[1]-query_patch[2]/2,query_patch[0]-query_patch[2]/2), (query_patch[1]+query_patch[2]/2,query_patch[0]+query_patch[2]/2), (255,0,0))
        if find_pos:
            cv2.rectangle(show_img1, (pos_patch[1]-pos_patch[2]/2,pos_patch[0]-pos_patch[2]/2), (pos_patch[1]+pos_patch[2]/2,pos_patch[0]+pos_patch[2]/2), COLOR[1])
        if find_neg:
            cv2.rectangle(show_img1, (neg_patch[1]-neg_patch[2]/2,neg_patch[0]-neg_patch[2]/2), (neg_patch[1]+neg_patch[2]/2,neg_patch[0]+neg_patch[2]/2), COLOR[0])
        seg_color0 = vlz_seg_color.convert_label_color_map(seg0).copy()
        seg_color1 = vlz_seg_color.convert_label_color_map(seg1).copy()
        cv2.imshow('Showing Patches0', show_img0)
        cv2.imshow('Showing Patches1', show_img1)
        cv2.imshow('Input seg0', seg_color0)
        cv2.imshow('Input seg1', seg_color1)
        cv2.waitKey(0)
    return output_blob

def createPatchImg(img, seg, npos, nneg, query_size, query_ignore, output_size, overlap_th1, overlap_th2, show=0): # img: CxHxW, seg: 1xHxW, query_size: better to be even
    '''
    INPUTS:
    img: HxWxC
    seg: HxWx1 or HxWx3
    npos: number of positive patches
    nneg: number of negative patches
    query_size: int, size of query patch
    query_ignore: query ignoring list
    output_size: int, size of output patches after resize
    overlap_th1: max overlap with positive for selecting negatives
    overlap_th2: min overlap with positive for selecting positives
    '''
    start_time = time.time()
    w = img.shape[1]
    h = img.shape[0]
    if len(seg.shape) != 3: 
        raise Exception('Seg should contain 3 dims.')
    if (seg.shape[0] != h or seg.shape[1] != w):
        raise Exception('img and seg size not consistent.')
    ## find query patch
    find_query_patch = False
    query_cls = -1
    for rand_id in range(100):
        query_center = [random.randint(query_size/2+1, h-query_size/2-1), random.randint(query_size/2+1, w-query_size/2-1)]
        for i in [i for i in range(21) if i not in query_ignore]:
            if computeOverlap(seg, query_center, query_size, i) > overlap_th2:
                find_query_patch = True
                query_cls = i
                break
        if find_query_patch:
            break
    if not find_query_patch:
        print('Cannot find a query patch.')
        return
    ## find positive patches
    npos_curr = 0
    nneg_curr = 0
    out_patches = np.zeros((npos+nneg, 4), dtype=int)
    total_iter = 0
    while (npos_curr < npos or nneg_curr < nneg):
        if total_iter > 1000:
            if npos_curr < npos:
                print('Hard to find enough positive patches.')
            if nneg_curr < nneg:
                print('Hard to find enough negative patches.')
            break
        total_iter += 1
        sample_center = [random.randint(0, h), random.randint(0, w)]
        for s in size_list:
            if not inSize(sample_center, s*h, h, w):
                continue
            ovlp = computeOverlap(seg, sample_center, s*h, query_cls)
            if (ovlp > overlap_th2 and npos_curr < npos):
                out_patches[npos_curr+nneg_curr,...] = np.array(sample_center + [int(s*float(h))] + [1])
                npos_curr += 1
                break
            elif (ovlp < overlap_th1 and nneg_curr < nneg and computeOverlap(seg, sample_center, s*h, 255) < overlap_th1):
                out_patches[npos_curr+nneg_curr,...] = np.array(sample_center + [int(s*float(h))] + [0])
                nneg_curr += 1
                break
    total_iter = 0
    while (npos_curr + nneg_curr < npos + nneg):
        if total_iter > 2000:
            print("Failed to find enough patches, exit.")
            sys.exit(1)
        total_iter += 1
        sample_center = [random.randint(0, h), random.randint(0, w)]
        for s in size_list:
            if not inSize(sample_center, s*h, h, w):
                continue
            ovlp = computeOverlap(seg, sample_center, s*h, query_cls)
            if (ovlp > overlap_th2):
                out_patches[npos_curr+nneg_curr,...] = np.array(sample_center + [int(s*float(h))] + [1])
                npos_curr += 1
                break
            elif (ovlp < overlap_th1 and computeOverlap(seg, sample_center, s*h, 255) < overlap_th1):
                out_patches[npos_curr+nneg_curr,...] = np.array(sample_center + [int(s*float(h))] + [0])
                nneg_curr += 1
                break
    # output blobs
    output_data_blob = np.zeros((npos+nneg+1, 3, output_size, output_size), dtype=np.float)
    output_cls_blob = np.zeros((npos+nneg), dtype=int)
    query_crop = img[query_center[0]-query_size/2:query_center[0]+query_size/2, query_center[1]-query_size/2:query_center[1]+query_size/2,:].copy()
    output_data_blob[0,...] = cv2.resize(query_crop, (output_size, output_size)).transpose((2,0,1)).astype(np.float)
    for i in range(npos+nneg):
        cx = out_patches[i,1]
        cy = out_patches[i,0]
        sz = out_patches[i,2]
        cls = out_patches[i,3]
        patch_data = img[cy-sz/2:cy+sz/2, cx-sz/2:cx+sz/2, :].copy()
        output_data_blob[i+1,...] = cv2.resize(patch_data, (output_size, output_size)).transpose((2,0,1)).astype(np.float)
        output_cls_blob[i] = cls
    #print("execution time: %s seconds" % (time.time() - start_time))
    # visualize
    if show:
        COLOR = [(0,0,255),(0,255,0)]
        show_img = img.copy()
        #print query_center
        #print query_size
        cv2.rectangle(show_img, (query_center[1]-query_size/2,query_center[0]-query_size/2), (query_center[1]+query_size/2,query_center[0]+query_size/2), (255,0,0))
        for i in range(nneg+npos):
            cx = out_patches[i,1]
            cy = out_patches[i,0]
            sz = out_patches[i,2]
            cls = out_patches[i,3]
            #print 'patch',i,cy,cx,sz,cls
            cv2.rectangle(show_img, (cx-sz/2,cy-sz/2), (cx+sz/2,cy+sz/2), COLOR[cls])
        seg_color = vlz_seg_color.convert_label_color_map(seg).copy()
        cv2.imshow('Showing Patches', show_img)
        cv2.imshow('Input seg', seg_color)
        cv2.waitKey(0)
    return output_data_blob, output_cls_blob

def main0():
    import time
    img_root = '/data/Imagenet/ILSVRC/Data/CLS-LOC/train/'
    seg_root = '/data/Imagenet/ILSVRC/Seg_infer/CLS-LOC/VGGrich_colorinit/'
    test_list = '/data/Imagenet/ILSVRC/ImageSets/CLS-LOC/imagenet_voc.txt'
    backend = '.JPEG'
    #img_root = '/data/VOCdevkit/VOC_arg/JPEGImages/'
    #seg_root = '/data/VOCdevkit/VOC_arg/SegmentationClass_label/'
    #test_list = '/data/VOCdevkit/VOC_arg/Lists/Raw/train.txt'
    with open(test_list, 'r') as infile:
        lines = infile.readlines()
    for idx in range(3000,len(lines)):
        line = lines[idx].rstrip('\n').split(' ')[0].split('.')[0]
        img_fn = img_root + line + backend
        seg_fn = seg_root + line + '.png'
        #print img_fn
        #print seg_fn
        img = cv2.imread(img_fn)
        if img is None:
            raise Exception("File not exist: %s" % img_fn)
        img_r = resize_uniform.resize_pad_to_fit(img, [480,480])
        seg = cv2.imread(seg_fn)
        if seg is None:
            raise Exception("File not exist: %s" % seg_fn)
        seg_r = resize_uniform.resize_pad_to_fit(seg, [480,480], pad_value=255, interp=cv2.INTER_NEAREST)
        print line
        start_time = time.time() 
        show_img = createPatchImg(img_r, seg_r, npos=1, nneg=3, query_size=128, query_ignore=[255], output_size=128, overlap_th1=0.15, overlap_th2=0.95, show=1)
        #cv2.imwrite('output/' + line + '.jpg', show_img)
def main1():
    import time
    img_root = '/data/Imagenet/ILSVRC/Data/CLS-LOC/train/'
    seg_root = '/data/Imagenet/ILSVRC/Seg_infer/CLS-LOC/VGGfcn_colorinit/'
    test_list = '/data/Imagenet/ILSVRC/ImageSets/CLS-LOC/Img/train_cls_JPEG_rand100k.txt'
    backend = '.JPEG'
    #img_root = '/data/VOCdevkit/VOC_arg/JPEGImages/'
    #seg_root = '/data/VOCdevkit/VOC_arg/SegmentationClass_label/'
    #test_list = '/data/VOCdevkit/VOC_arg/Lists/Raw/train.txt'
    with open(test_list, 'r') as infile:
        lines = infile.readlines()
#    outfile = open(valid_list, 'w')
#    outfile_fail = open(fail_list, 'w')
    for idx in range(0,len(lines)):
        line = lines[idx].rstrip('\n').split(' ')[0].split('.')[0]
        img_fn = img_root + line + backend
        seg_fn = seg_root + line + '.png'
        img = cv2.imread(img_fn)
        if img is None:
            raise Exception("File not exist: %s" % img_fn)
        img_r = resize_uniform.resize_pad_to_fit(img, [480,480])
        seg = cv2.imread(seg_fn)
        if seg is None:
            raise Exception("File not exist: %s" % seg_fn)
        seg_r = resize_uniform.resize_pad_to_fit(seg, [480,480], pad_value=255, interp=cv2.INTER_NEAREST)
        createTripletPatchImg(img_r, seg_r, query_max_portion=False, query_ignore=[255], output_size=128, overlap_th1=0.2, overlap_th2=0.85, pos_bbox_iou=0.4, mode=1, show=1)
#        if state:
#            outfile.write(lines[idx])
#        else:
#            outfile_fail.write(lines[idx])
#    outfile.close()
#    outfile_fail.close()

def main2():
    import time
    img_root0 = '/data/VOCdevkit/VOC_arg/JPEGImages/'
    img_root1 = '/data/Imagenet/ILSVRC/Data/CLS-LOC/train/'
    seg_root0 = '/data/VOCdevkit/results/VOC_arg/Seg_infer/VGGfcn_colorinit_train_iter_130000/'
    seg_root1 = '/data/Imagenet/ILSVRC/Seg_infer/CLS-LOC/VGGrich_colorinit/'
    test_list0 = '/data/VOCdevkit/VOC_arg/Lists/Raw/train.txt'
    test_list1 = '/data/Imagenet/ILSVRC/ImageSets/CLS-LOC/imagenet_voc/imagenet_fc7_top2.txt'
    with open(test_list0, 'r') as infile:
        lines0 = infile.readlines()
    with open(test_list1, 'r') as infile:
        lines1 = infile.readlines()
    for idx in range(0,len(lines0)):
        line0 = lines0[idx].rstrip('\n').split(' ')[0].split('.')[0]
        line1 = lines1[2*idx].rstrip('\n').split(' ')[0].split('.')[0]
        img_fn0 = img_root0 + line0 + '.jpg'
        img_fn1 = img_root1 + line1 + '.JPEG'
        seg_fn0 = seg_root0 + line0 + '.png'
        seg_fn1 = seg_root1 + line1 + '.png'
        img0 = cv2.imread(img_fn0)
        img1 = cv2.imread(img_fn1)
        if img0 is None:
            raise Exception("File not exist: %s" % img_fn0)
        if img1 is None:
            raise Exception("File not exist: %s" % img_fn1)
        img_r0 = resize_uniform.resize_pad_to_fit(img0, [480,480])
        img_r1 = resize_uniform.resize_pad_to_fit(img1, [480,480])
        seg0 = cv2.imread(seg_fn0)
        seg1 = cv2.imread(seg_fn1)
        if seg0 is None:
            raise Exception("File not exist: %s" % seg_fn0)
        if seg1 is None:
            raise Exception("File not exist: %s" % seg_fn1)
        seg_r0 = resize_uniform.resize_pad_to_fit(seg0, [480,480], pad_value=255, interp=cv2.INTER_NEAREST)
        seg_r1 = resize_uniform.resize_pad_to_fit(seg1, [480,480], pad_value=255, interp=cv2.INTER_NEAREST)
        createTripletPatchTwoImg(img_r0, img_r1, seg_r0, seg_r1, query_max_portion=True, fix_query=cls, query_ignore=[0,255], output_size=128, overlap_th1=0.2, overlap_th2=0.85, mode=1, show=1)

def main3():
    import os
    #img_root = '/data/VOCdevkit/VOC_arg/JPEGImages/'
    #seg_root = '/data/VOCdevkit/VOC_arg/SegmentationClass_label/'
    #test_list = '/data/VOCdevkit/VOC_arg/Lists/Img+Cls/train_maxCls.txt'
    #backend = '.jpg'
    img_root = '/data/Imagenet/ILSVRC/Data/CLS-LOC/train/'
    seg_root = '/data/Imagenet/ILSVRC/Seg_infer/CLS-LOC/VGGfcn_colorinit/'
    test_list = '/data/Imagenet/ILSVRC/ImageSets/CLS-LOC/imagenet_voc/imagenet_voc_maxCls.txt'
    out_root = '/data/Imagenet/ILSVRC/Data/CLS-LOC/patches/Shared_VOC_fcn_colorinit/'
    backend = '.JPEG'
    for i in range(21):
        if not os.path.isdir(out_root + '/' + str(i)):
            os.makedirs(out_root + '/' + str(i))
    with open(test_list, 'r') as infile:
        lines = infile.readlines()
    start_time = time.time()
    for idx in range(70000, len(lines)):
        if idx%100 == 0:
            print('Processing: %d/%d, Total time: %s seconds' % (idx, len(lines), time.time() - start_time))
        line = lines[idx].rstrip('\n').split(' ')[0]
        write_fn = line.split('/')[-1]
        cls = int(lines[idx].rstrip('\n').split(' ')[1])
        img_fn = img_root + line + backend
        seg_fn = seg_root + line + '.png'
        img = resize_uniform.resize_pad_to_fit(cv2.imread(img_fn), [480,480], pad_value=0)
        seg = resize_uniform.resize_pad_to_fit(cv2.imread(seg_fn), [480,480], pad_value=255, interp=cv2.INTER_NEAREST)
        if img is None:
            raise Exception("File not exist: %s" % img_fn)
        if seg is None:
            raise Exception("File not exist: %s" % seg_fn)
        pos_valid, pos_patches = createSinglePatchImg(img, seg, query=cls, num=5, output_size=128, overlap_th=0.8, pos_bbox_iou=0.6, show=0)
        neg_valid, neg_patches = createSinglePatchImg(img, seg, query=0, num=5, output_size=128, overlap_th=0.9, pos_bbox_iou=0.45, show=0)
        # write
        for v in range(pos_valid):
            patch = img[pos_patches[v,0]-pos_patches[v,2]/2:pos_patches[v,0]+pos_patches[v,2]/2, pos_patches[v,1]-pos_patches[v,2]/2:pos_patches[v,1]+pos_patches[v,2]/2, :]
            patch_r = cv2.resize(patch, (128, 128))
#            print('write to: %s' % (out_root + '/' + str(cls) + '/' + write_fn + '_' + str(v) + '.jpg'))
            cv2.imwrite(out_root + '/' + str(cls) + '/' + write_fn + '_' + str(v) + '.jpg', patch_r)
        for v in range(neg_valid):
            patch = img[neg_patches[v,0]-neg_patches[v,2]/2:neg_patches[v,0]+neg_patches[v,2]/2, neg_patches[v,1]-neg_patches[v,2]/2:neg_patches[v,1]+neg_patches[v,2]/2, :]
            patch_r = cv2.resize(patch, (128, 128))
#            print('write to: %s' % (out_root + '/0/' + write_fn + '_' + str(v) + '.jpg'))
            cv2.imwrite(out_root + '/0/' + write_fn + '_' + str(v) + '.jpg', patch_r)
if __name__ == "__main__":
    main1()

