import cv2
import numpy as np

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
