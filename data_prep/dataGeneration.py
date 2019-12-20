# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""

import glob
import os
import shutil

baseNames = []
sourceFileNames = []
destfiles = []
sourceFilepath =   "FinalExtensiveDataMerged/BoundingBoxesNewBackground/NewBackground/selectedNewBackground/" #'FinalExtensiveData/mug-bb/'
fileList = glob.glob('FinalExtensiveDataMerged/BoundingBoxesNewBackground/NewBackground/selectedNewBackground/*.png')
destinationPath =   'FinalExtensiveDataMerged/BackgroundwithHomeLab/' #'FinalExtensiveData/mug-bb-rotated/'


for files in fileList:
    base = os.path.basename(files)
    fileName = os.path.splitext(base)[0]
    baseNames.append(base)
    sourceFileNames.append(sourceFilepath + base)
    destfiles.append(destinationPath + 'Homelab' + base)
    try:
        shutil.copy(sourceFilepath + base, destinationPath + 'Homelab' + base)
    except:
        print 'file does not exist'
        continue
