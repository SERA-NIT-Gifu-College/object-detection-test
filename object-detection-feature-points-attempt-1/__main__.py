import cv2
import os
import numpy as np

size_rec = (640, 850)
akaze = cv2.AKAZE_create()
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

query_image = cv2.imread(os.path.join(os.getcwd(), 'data/cone.jpg'))
query_image = cv2.resize(query_image, size_rec)
query_image_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
query_image_key_points = orb.detect(query_image)
result_image = cv2.drawKeypoints(query_image, query_image_key_points, None, flags=4)

query_video = cv2.VideoCapture(os.path.join(os.getcwd(), 'data/video.mp4'))
# query_video = cv2.VideoCapture(0)

query_image_kp1, dest1 = orb.detectAndCompute(query_image, None)

while (1):
    ret, tmp = query_video.read()
    if not ret:
        query_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    src = cv2.resize(tmp, size_rec)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    query_video_frame_key_points = orb.detect(src)
    result_frame = cv2.drawKeypoints(src, query_video_frame_key_points, None, flags=4)
    query_video_frame_kp, dest_frame = orb.detectAndCompute(src, None)
    matches = bf.match(dest1, dest_frame)
    # matches = sorted(matches, key = lambda x:x.distance)
    good_matches = [ [good] for good in matches if good.distance < 30 ]
    match_result = cv2.drawMatchesKnn(query_image, query_image_kp1, src, query_video_frame_kp, good_matches, None)
    cv2.imshow('feature points (frame)', result_frame)
    cv2.imshow('feature points image', result_image)
    cv2.imshow('feature points matches', match_result)
    similarity = len(good_matches) / len(query_video_frame_kp)
    print(f"Similarity: {similarity}")
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
query_video.release()
