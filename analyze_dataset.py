import pandas as pd
import cv2
from pathlib import Path

# MOT17 format: frame_id, track_id, x, y, w, h, confidence, class_id, visibility
gt_df = pd.read_csv('MOT17-02-FRCNN/gt/gt.txt', header=None)
gt_df.columns = ['frame', 'track_id', 'x', 'y', 'w', 'h', 'confidence', 'class_id', 'visibility']

print('=' * 60)
print('MOT17-02-FRCNN DATASET ANALYSIS')
print('=' * 60)
print(f'\nTotal detections: {len(gt_df)}')
print(f'Frame range: {int(gt_df["frame"].min())} - {int(gt_df["frame"].max())}')
print(f'Total unique tracks (objects): {gt_df["track_id"].nunique()}')

print(f'\n--- CLASS DISTRIBUTION ---')
print(gt_df['class_id'].value_counts().sort_index())

print(f'\n--- MOT17 CLASS MAPPING ---')
class_map = {
    1: 'Pedestrian',
    2: 'Person on vehicle',
    3: 'Car',
    4: 'Bicycle',
    5: 'Motorbike',
    6: 'Non-motorized vehicle',
    7: 'Static person',
    8: 'Distractor',
    9: 'Occluded pedestrian',
    10: 'Crowd'
}
for class_id in sorted(gt_df['class_id'].unique()):
    count = len(gt_df[gt_df['class_id'] == class_id])
    class_name = class_map.get(class_id, f'Unknown')
    print(f'  Class {class_id}: {class_name:30s} - {count:6d} detections')

print(f'\n--- ISSUE DETECTED ---')
print(f'Your code filters for: class_id = 0 (COCO person)')
print(f'GT file contains:      class_id = 1,7,8,9,10 (MOT17 classes)')
print(f'\nMOT17 vs COCO class mismatch!')
print(f'  MOT17 class 1  = Pedestrian (should match COCO class 0)')
print(f'  MOT17 class 7  = Static person')
print(f'  MOT17 class 8  = Distractor')
print(f'  MOT17 class 9  = Occluded pedestrian')
print(f'\n>>> Solution: Change filter to classes=[1,7,9] in YOLO tracking')

# Check image content
img_path = 'MOT17-02-FRCNN/img1/000001.jpg'
img = cv2.imread(img_path)
if img is not None:
    print(f'\n--- IMAGE INFO ---')
    print(f'Image size: {img.shape}')
    print(f'Sample frame has {len(gt_df[gt_df["frame"]==1])} GT detections')
else:
    print('Image not found!')
