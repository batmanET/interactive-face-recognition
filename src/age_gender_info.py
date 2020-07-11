"""
Created on Wed Jul 8 16:42:22 2020

@author: ARawat4
"""

import numpy as np

from .utils import cut_rois, resize_input
from .ie_module import Module

class AgeGenderDetector(Module):
    class Result:
        def __init__(self, age_output, gender_output):
            self.age = age_output
            self.gender = "Male" if gender_output[0][1][0][0]>gender_output[0][0][0][0] else "Female"
            self.info = f"{self.gender}, {str(int(age_output[0][0][0][0] * 100))} years"
        

    def __init__(self, model):
        super(AgeGenderDetector, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 2, "Expected 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.age_gender_iter = iter(model.outputs)
        self.age_out_blob = next(self.age_gender_iter)
        self.gender_out_blob = next(self.age_gender_iter)
        self.input_shape = model.inputs[self.input_blob].shape

    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(AgeGenderDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_age(self):
        outputs = self.get_outputs()
        results = [AgeGenderDetector.Result(out[self.age_out_blob], out[self.gender_out_blob]) \
                      for out in outputs]
        return results
