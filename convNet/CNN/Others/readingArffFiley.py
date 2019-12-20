# -*- coding: utf-8 -*-

import arff
import numpy

data = arff.load(open('/export/pablo/Datasets/Val_AFEW_5_0/Audio/ExpressionWise/Angry/000149120.arff', 'rb'))


print numpy.array(data["data"]).shape

