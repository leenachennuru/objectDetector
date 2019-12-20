# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""

import numpy as np
import cv2
import shutil as copy

f = open('TrashcanBackBB.txt', 'r')
sourceDir = "FirstScenarioDataset/TrashcanBB/"
destDir = "FirstScenarioDataset/trashcanBackgroundclassBB/"
count =0
for line in f:
	abcLine = line.split("\n")[0] + ".png"
	copy.move(sourceDir + abcLine, destDir + abcLine)
	count = count+1
