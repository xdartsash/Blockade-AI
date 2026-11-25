import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from overlay_boxes import find_boxes,stable_boxes,erase_proposals,yolo_lines

class OverlayTests(unittest.TestCase):
    def frame(self): return np.full((200,240,3),30,np.uint8)

    def test_rectangle_and_color_do_not_mean_team(self):
        a=self.frame()
        cv2.rectangle(a,(40,30),(80,120),(0,255,0),1)
        boxes=find_boxes(a)
        self.assertEqual(len(boxes),1)
        self.assertIsNone(boxes[0]['class_id'])
        self.assertEqual(boxes[0]['xyxy'],[40,30,81,121])

    def test_colored_wall_and_open_outline_rejected(self):
        for color in [(0,255,0),(0,0,255)]:
            a=self.frame()
            cv2.rectangle(a,(40,30),(80,120),color,-1)
            self.assertEqual(find_boxes(a),[])
            a=self.frame()
            cv2.line(a,(40,30),(40,120),color,1)
            cv2.line(a,(40,120),(80,120),color,1)
            self.assertEqual(find_boxes(a),[])

    def test_red_never_exported(self):
        a=self.frame()
        cv2.rectangle(a,(40,30),(80,120),(0,0,255),1)
        boxes=find_boxes(a)
        self.assertEqual(len(boxes),1)
        self.assertEqual(yolo_lines(boxes,a.shape,0),'')

    def test_temporal_confirmation(self):
        a=self.frame()
        cv2.rectangle(a,(40,30),(80,120),(0,255,0),1)
        boxes=find_boxes(a)
        self.assertEqual(stable_boxes(boxes,[]),[])
        self.assertEqual(stable_boxes(boxes,boxes),boxes)

    def test_cleanup_preserves_original_and_center(self):
        a=self.frame()
        cv2.rectangle(a,(40,30),(80,120),(0,255,0),1)
        original=a.copy()
        clean,mask=erase_proposals(a,find_boxes(a))
        np.testing.assert_array_equal(a,original)
        np.testing.assert_array_equal(clean[50:100,50:70],a[50:100,50:70])
        self.assertGreater(mask.sum(),0)

    def test_user_crops(self):
        fixtures=Path(__file__).parent/'fixtures'
        for name,color,xyxy in [('green.png','green',[55,47,111,157]),('red.png','red',[41,40,62,81])]:
            a=cv2.imdecode(np.frombuffer((fixtures/name).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            boxes=find_boxes(a)
            self.assertEqual(len(boxes),1)
            self.assertEqual(boxes[0]['color'],color)
            self.assertEqual(boxes[0]['xyxy'],xyxy)

if __name__=='__main__': unittest.main()
