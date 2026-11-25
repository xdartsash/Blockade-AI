"""Conservative detection of thin RGB ESP rectangles; never identifies a team.

All output is an unreviewed proposal. Matching geometry can exist in the world.
"""
import cv2
import numpy as np


def possible_nameplate(frame, xyxy):
    """White glyph-like components above a box, NOT OCR or team recognition."""
    x1,y1,x2,y2=xyxy
    w,h=x2-x1,y2-y1
    left=max(0,x1-w)
    right=min(frame.shape[1],x2+w)
    top=max(0,y1-min(50,max(14,int(h*.4))))
    bottom=min(frame.shape[0],y1+max(2,int(h*.1)))
    roi=frame[top:bottom,left:right]
    if roi.size==0:
        return False
    low=roi.min(axis=2)
    high=roi.max(axis=2)
    mask=((low>155)&((high.astype(np.int16)-low)<55)).astype(np.uint8)*255
    _,_,stats,centers=cv2.connectedComponentsWithStats(mask)
    glyphs=[(s,c) for s,c in zip(stats[1:],centers[1:])
            if 2<=s[4]<=140 and 2<=s[2]<=30 and 3<=s[3]<=22]
    for _,center in glyphs:
        row=[s for s,c in glyphs if abs(c[1]-center[1])<=4]
        if len(row)<3:
            continue
        start=min(s[0] for s in row)
        end=max(s[0]+s[2] for s in row)
        row_center=left+(start+end)/2
        if end-start>=12 and abs(row_center-(x1+x2)/2)<=w:
            return True
    return False


def find_boxes(frame):
    b, g, r = cv2.split(frame)
    masks = {
        # Live antialiasing makes some edges darker than the initial crops.
        # Require dominant saturated color, not brightness 220 on every edge.
        'green': ((g >= 175) & (r <= 60) & (b <= 60)
                  & (g.astype(np.int16)-np.maximum(r,b)>=130)).astype(np.uint8)*255,
        'red': ((r >= 175) & (g <= 60) & (b <= 60)
                & (r.astype(np.int16)-np.maximum(g,b)>=130)).astype(np.uint8)*255,
    }
    results = []
    for color, mask in masks.items():
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for x, y, w, h, area in stats[1:]:
            x,y,w,h,area = map(int,(x,y,w,h,area))
            if not (8 <= w and 16 <= h and 1.15 <= h/w <= 4.5):
                continue
            # Clipped outlines cannot be checked on all four sides.
            if x == 0 or y == 0 or x+w >= frame.shape[1] or y+h >= frame.shape[0]:
                continue
            roi = mask[y:y+h,x:x+w] > 0
            # A few antialiased fringe pixels must not move an entire side
            # outwards and make a genuine closed outline look open.
            columns=np.flatnonzero(roi.mean(axis=0)>=.65)
            rows=np.flatnonzero(roi.mean(axis=1)>=.65)
            if len(columns)<2 or len(rows)<2:
                continue
            dx,dy=int(columns[0]),int(rows[0])
            w,h=int(columns[-1]-dx+1),int(rows[-1]-dy+1)
            x,y=x+dx,y+dy
            if not (w>=8 and h>=16 and 1.15<=h/w<=4.5):
                continue
            roi=mask[y:y+h,x:x+w]>0
            area=int(roi.sum())
            t = min(3,max(1,round(min(w,h)*.025)))
            coverage = [roi[:t,:].any(axis=0).mean(), roi[-t:,:].any(axis=0).mean(),
                        roi[:,:t].any(axis=1).mean(), roi[:,-t:].any(axis=1).mean()]
            inside = roi[t:-t,t:-t]
            interior_density = float(inside.mean()) if inside.size else 1.
            if min(coverage) < .9 or interior_density > .04:
                continue
            # Occupancy near the theoretical perimeter rejects thick blocks.
            if area > 2*(w+h)*3:
                continue
            results.append({'xyxy':[x,y,x+w,y+h], 'color':color,
                            'edge_coverage':float(min(coverage)),
                            'class_id':None, 'review_required':True,
                            'possible_nameplate':possible_nameplate(frame,[x,y,x+w,y+h])})
    return sorted(results,key=lambda box:(box['xyxy'][0],box['xyxy'][1]))


def iou(a,b):
    x1,y1,x2,y2=a
    u1,v1,u2,v2=b
    inter=max(0,min(x2,u2)-max(x1,u1))*max(0,min(y2,v2)-max(y1,v1))
    return inter/max(1,(x2-x1)*(y2-y1)+(u2-u1)*(v2-v1)-inter)


def stable_boxes(current, previous):
    return [box for box in current if any(
        box['color']==old['color'] and iou(box['xyxy'],old['xyxy']) >= .5
        for old in previous)]


def erase_proposals(frame, boxes):
    """Approximation only; removes a narrow perimeter band, may damage pixels."""
    mask=np.zeros(frame.shape[:2],np.uint8)
    for box in boxes:
        x1,y1,x2,y2=box['xyxy']
        cv2.rectangle(mask,(x1,y1),(x2-1,y2-1),255,3)
    return cv2.inpaint(frame,mask,3,cv2.INPAINT_TELEA),mask


def yolo_lines(boxes, shape, class_id):
    if class_id not in (0,1):
        raise ValueError('Class must be assigned explicitly: 0 enemy, 1 teammate')
    height,width=shape[:2]
    lines=[]
    for box in boxes:
        if box['color']!='green':
            continue
        x1,y1,x2,y2=box['xyxy']
        lines.append(f'{class_id} {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}')
    return '\n'.join(lines)+'\n' if lines else ''
