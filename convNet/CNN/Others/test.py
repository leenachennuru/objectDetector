# -*- coding: utf-8 -*-

import cv2

img1 = cv2.imread("/data/datasets/SAVEE/POW_SPEC_2048_1/DC/a/00.png")

img2 = cv2.imread("/data/datasets/SAVEE/ALL/POW_TEST_2048_1/DC/a/00.png")


print img1

print img2


img3 = img1-img2

print img3

cv2.imwrite("/data/datasets/SAVEE/test.png", img3)