"""
Created on Wed Jul 8 13:01:53 2020

@author: ARawat4
"""

import numpy as np

from .utils import cut_rois, resize_input
from .ie_module import Module

class EmotionDetector(Module):
    class Result:
        def __init__(self, output):
            self.emotions = {"Neutral": round(output[0][0][0][0]*100, 1), "Happy": round(output[0][1][0][0]*100, 1), \
                                "Sad": round(output[0][2][0][0]*100, 1), "Surprise": round(output[0][3][0][0]*100, 1), \
                                "Anger": round(output[0][4][0][0]*100, 1)}
            self.emotion = sorted(self.emotions.items(), key=lambda x: x[1], reverse=True)[0]
            self.emotion_text = "{}: {}".format(str(self.emotion[0]), str(self.emotion[1]))
        

    def __init__(self, model):
        super(EmotionDetector, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape

    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(EmotionDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_emotions(self):
        outputs = self.get_outputs()
        results = [EmotionDetector.Result(out[self.output_blob]) \
                      for out in outputs]
        return results
