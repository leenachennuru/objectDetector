import os
from pylearn2.utils import serial
import pylearn2.config.yaml_parse as yaml_parse

in_path = "params_experiment_test1_Epoch_15.save"
out_path = "GPU_params_experiment_test1_Epoch_15.save"



os.environ['THEANO_FLAGS']="device=cpu"

model = serial.load(in_path)

model2 = yaml_parse.load(model.yaml_src)
model2.set_param_values(model.get_param_values())

serial.save(out_path, model2)