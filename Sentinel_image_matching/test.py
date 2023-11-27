import os
import cv2
import matplotlib.pyplot as plt
import config
from utils import RemoveBlackFilling


img = cv2.imread(os.path.join(config.data_path, config.file_name))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = RemoveBlackFilling(img)

descriptor = cv2.SIFT_create()
img_features = descriptor.detectAndCompute(img, None)

input_img_paths = [f for f in os.listdir(
    config.data_path) if f.endswith(".jpg")]

for img_path in input_img_paths:
    if img_path == config.file_name:
        continue
    img_match = cv2.imread(os.path.join(config.data_path, img_path))
    img_match = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)
    img_match = RemoveBlackFilling(img_match)

    img_match_features = descriptor.detectAndCompute(img_match, None)

    # If there are a lot of key points, it may mean that the image is very noisy
    if len(img_match_features[1]) > 7000:
        continue

    # Brute-force matcher with cross-check
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(img_features[1], img_match_features[1])

    # skip images with low numbers of matches
    if len(matches) < 1000:
        continue

    matches = sorted(matches, key=lambda x: x.distance)

    # draw 100 best matches
    result = cv2.drawMatches(img, img_features[0], img_match, img_match_features[0], matches[:100],
                             None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure()
    plt.imshow(result)
    plt.title(f"{img_path}       total matches: {len(matches)}")

plt.show()
