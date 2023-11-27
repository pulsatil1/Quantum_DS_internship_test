import os
import cv2
import matplotlib.pyplot as plt
import config
from utils import RemoveBlackFilling


input_img_paths = [f for f in os.listdir(
    config.data_path) if f.endswith(".jpg")]

img_src = cv2.imread(os.path.join(config.data_path, input_img_paths[10]))
img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
img_src = RemoveBlackFilling(img_src)

img_dst = cv2.imread(os.path.join(config.data_path, input_img_paths[11]))
img_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2RGB)
img_dst = RemoveBlackFilling(img_dst)

# SIFT detector
descriptor = cv2.SIFT_create()

(kps_src, features_src) = descriptor.detectAndCompute(img_src, None)
(kps_dst, features_dst) = descriptor.detectAndCompute(img_dst, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(features_src, features_dst)

matches = sorted(matches, key=lambda x: x.distance)

# draw 100 best matches
result = cv2.drawMatches(img_src, kps_src, img_dst, kps_dst, matches[:100],
                         None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(result)
plt.show()
