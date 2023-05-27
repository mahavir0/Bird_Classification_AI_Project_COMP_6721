import cv2, sys, os
fpath = sys.argv[1]
if os.path.exists(fpath):
    cv2.imwrite(fpath, cv2.imread(fpath))