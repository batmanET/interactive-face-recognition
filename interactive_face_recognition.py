#!/usr/bin/env python

"""
Created on Wed Jul 8 13:01:10 2020

@author: ARawat4
"""

import logging as log
import os.path as osp
import sys
import time
from argparse import ArgumentParser

import cv2
import numpy as np

from openvino.inference_engine import IENetwork
from src.ie_module import InferenceContext
from src.landmarks import LandmarksDetector
from src.age_gender_info import AgeGenderDetector
from src.emotion_detection import EmotionDetector
from src.face_detection import FaceDetector
from src.high_angle_person import HighAnglePersonDetector
from src.faces_database import FacesDatabase
from src.reidentification import FaceIdentifier

DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']
MATCH_ALGO = ['HUNGARIAN', 'MIN_DIST']


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', metavar="PATH", default='0',
                         help="(optional) Path to the input video " \
                         "('0' for the camera, default)")
    general.add_argument('-o', '--output', metavar="PATH", default="",
                         help="(optional) Path to save the output video to")
    general.add_argument('--no_show', action='store_true',
                         help="(optional) Do not display output")
    general.add_argument('-tl', '--timelapse', action='store_true',
                         help="(optional) Auto-pause after each frame")
    general.add_argument('-cw', '--crop_width', default=0, type=int,
                         help="(optional) Crop the input stream to this width " \
                         "(default: no crop). Both -cw and -ch parameters " \
                         "should be specified to use crop.")
    general.add_argument('-ch', '--crop_height', default=0, type=int,
                         help="(optional) Crop the input stream to this height " \
                         "(default: no crop). Both -cw and -ch parameters " \
                         "should be specified to use crop.")
    general.add_argument('--match_algo', default='HUNGARIAN', choices=MATCH_ALGO,
                         help="(optional)algorithm for face matching(default: %(default)s)")

    gallery = parser.add_argument_group('Faces database')
    gallery.add_argument('-fg', metavar="PATH", default="face_db",
                         help="Path to the face images directory")
    gallery.add_argument('--run_detector', action='store_true',
                         help="(optional) Use Face Detection model to find faces" \
                         " on the face images, otherwise use full images.")

    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', metavar="PATH", default="models/face-detection-retail-0004.xml",
                        help="Path to the Face Detection model XML file")
    models.add_argument('-m_pd', metavar="PATH", default="models/person-detection-retail-0013.xml",
                        help="Path to the High Angle Person Detection model XML file")
    models.add_argument('-m_lm', metavar="PATH", default="models/landmarks-regression-retail-0009.xml",
                        help="Path to the Facial Landmarks Regression model XML file")
    models.add_argument('-m_ag', metavar="PATH", default="models/age-gender-recognition-retail-0013.xml",
                        help="Path to the Age and Gender Detection model XML file")
    models.add_argument('-m_ed', metavar="PATH", default="models/emotions-recognition-retail-0003.xml",
                        help="Path to the Emotion Detection model XML file")
    models.add_argument('-m_reid', metavar="PATH", default="models/face-reidentification-retail-0095.xml",
                        help="Path to the Face Reidentification model XML file")
    models.add_argument('-fd_iw', '--fd_input_width', default=0, type=int,
                         help="(optional) specify the input width of detection model " \
                         "(default: use default input width of model). Both -fd_iw and -fd_ih parameters " \
                         "should be specified for reshape.")
    models.add_argument('-fd_ih', '--fd_input_height', default=0, type=int,
                         help="(optional) specify the input height of detection model " \
                         "(default: use default input height of model). Both -fd_iw and -fd_ih parameters " \
                         "should be specified for reshape.")
    
    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Detection model (default: %(default)s)")
    infer.add_argument('-d_pd', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "High Angle Person Detection model (default: %(default)s)")
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Facial Landmarks Regression model (default: %(default)s)")
    infer.add_argument('-d_ag', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Age and Gender Detection model (default: %(default)s)")
    infer.add_argument('-d_ed', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Emotion Detection model (default: %(default)s)")
    infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Reidentification model (default: %(default)s)")
    infer.add_argument('-l', '--cpu_lib', metavar="PATH", default="",
                       help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. " \
                       "Path to a shared library with custom layers implementations")
    infer.add_argument('-c', '--gpu_lib', metavar="PATH", default="",
                       help="(optional) For clDNN (GPU)-targeted custom layers, if any. " \
                       "Path to the XML file with descriptions of the kernels")
    infer.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    infer.add_argument('-pc', '--perf_stats', action='store_true',
                       help="(optional) Output detailed per-layer performance stats")
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
                       help="(optional) Probability threshold for face detections" \
                       "(default: %(default)s)")
    infer.add_argument('-t_pd', metavar='[0..1]', type=float, default=0.8,
                       help="(optional) Probability threshold for high angle person detections" \
                       "(default: %(default)s)")
    infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
                       help="(optional) Cosine distance threshold between two vectors " \
                       "for face identification (default: %(default)s)")
    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help="(optional) Scaling ratio for bboxes passed to face recognition " \
                       "(default: %(default)s)")
    infer.add_argument('-exp_r_pd', metavar='NUMBER', type=float, default=1,
                       help="(optional) Scaling ratio for bboxes passed to high angle person recognition " \
                       "(default: %(default)s)")
    infer.add_argument('--allow_grow', action='store_true',
                       help="(optional) Allow to grow faces gallery and to dump on disk. " \
                       "Available only if --no_show option is off.")

    return parser



class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        used_devices = set([args.d_fd, args.d_pd, args.d_lm, args.d_ag, args.d_ed, args.d_reid])
        self.context = InferenceContext()
        context = self.context
        context.load_plugins(used_devices, args.cpu_lib, args.gpu_lib)
        for d in used_devices:
            context.get_plugin(d).set_config({
                "PERF_COUNT": "YES" if args.perf_stats else "NO"})

        log.info("Loading models")
        face_detector_net = self.load_model(args.m_fd)
        high_angle_person_detector_net = self.load_model(args.m_pd)
        
        assert (args.fd_input_height and args.fd_input_width) or \
               (args.fd_input_height==0 and args.fd_input_width==0), \
            "Both -fd_iw and -fd_ih parameters should be specified for reshape"
        
        if args.fd_input_height and args.fd_input_width :
            face_detector_net.reshape({"data": [1, 3, args.fd_input_height,args.fd_input_width]})
            high_angle_person_detector_net.reshape({"data": [1, 3, args.fd_input_height,args.fd_input_width]})

        landmarks_net = self.load_model(args.m_lm)
        age_gender_net = self.load_model(args.m_ag)
        emotion_net = self.load_model(args.m_ed)
        face_reid_net = self.load_model(args.m_reid)

        self.face_detector = FaceDetector(face_detector_net,
                                          confidence_threshold=args.t_fd,
                                          roi_scale_factor=args.exp_r_fd)
        
        self.high_angle_person_detector = HighAnglePersonDetector(high_angle_person_detector_net,
                                          confidence_threshold=args.t_pd,
                                          roi_scale_factor=args.exp_r_pd)

        self.landmarks_detector = LandmarksDetector(landmarks_net)
        self.age_detector = AgeGenderDetector(age_gender_net)
        self.emotion_detector = EmotionDetector(emotion_net)
        self.face_identifier = FaceIdentifier(face_reid_net,
                                              match_threshold=args.t_id,
                                              match_algo = args.match_algo)

        self.face_detector.deploy(args.d_fd, context)
        self.high_angle_person_detector.deploy(args.d_pd, context)
        self.landmarks_detector.deploy(args.d_lm, context,
                                       queue_size=self.QUEUE_SIZE)
        self.age_detector.deploy(args.d_ag, context,
                                    queue_size=self.QUEUE_SIZE)
        self.emotion_detector.deploy(args.d_ed, context,
                                    queue_size=self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, context,
                                    queue_size=self.QUEUE_SIZE)
        log.info("Models are loaded")

        log.info("Building faces database using images from '%s'" % (args.fg))
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if args.run_detector else None, args.no_show)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info("Database is built, registered %s identities" % \
            (len(self.faces_database)))

        self.allow_grow = args.allow_grow and not args.no_show

    def load_model(self, model_path):
        model_path = osp.abspath(model_path)
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        log.info("Loading the model from '%s'" % (model_path))
        assert osp.isfile(model_path), \
            "Model description is not found at '%s'" % (model_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
        model = IENetwork(model_path, model_weights_path)
        log.info("Model is loaded")
        return model

    def process(self, frame):
        assert len(frame.shape) == 3, \
            "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], \
            "Expected BGR or BGRA input"

        orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        frame = np.expand_dims(frame, axis=0)

        self.face_detector.clear()
        self.high_angle_person_detector.clear()
        self.landmarks_detector.clear()
        self.age_detector.clear()
        self.emotion_detector.clear()
        self.face_identifier.clear()

        
        self.high_angle_person_detector.start_async(frame)
        person_rois = self.high_angle_person_detector.get_roi_proposals(frame)
        self.face_detector.start_async(frame)
        rois = self.face_detector.get_roi_proposals(frame)
        if self.QUEUE_SIZE < len(rois):
            log.warning("Too many faces for processing." \
                    " Will be processed only %s of %s." % \
                    (self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]
        self.landmarks_detector.start_async(frame, rois)
        landmarks = self.landmarks_detector.get_landmarks()

        self.age_detector.start_async(frame, rois)
        age_genders = self.age_detector.get_age()

        self.emotion_detector.start_async(frame, rois)
        emotions = self.emotion_detector.get_emotions()

        self.face_identifier.start_async(frame, rois, landmarks)
        face_identities, unknowns = self.face_identifier.get_matches()
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                    (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop = orig_image[int(rois[i].position[1]):int(rois[i].position[1]+rois[i].size[1]), int(rois[i].position[0]):int(rois[i].position[0]+rois[i].size[0])]
                name = self.faces_database.ask_to_save(crop)
                if name:
                    id = self.faces_database.dump_faces(crop, face_identities[i].descriptor, name)
                    face_identities[i].id = id

        outputs = [[rois, landmarks, face_identities, age_genders, emotions], person_rois]

        return outputs


    def get_performance_stats(self):
        stats = {
            'face_detector': self.face_detector.get_performance_stats(),
            'landmarks': self.landmarks_detector.get_performance_stats(),
            'face_identifier': self.face_identifier.get_performance_stats(),
        }
        return stats


class Visualizer:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}


    def __init__(self, args):
        self.frame_processor = FrameProcessor(args)
        self.display = not args.no_show
        self.print_perf_stats = args.perf_stats

        self.frame_time = 0
        self.frame_start_time = time.perf_counter()
        self.fps = 0
        self.frame_num = 0
        self.frame_count = -1

        self.input_crop = None
        if args.crop_width and args.crop_height:
            self.input_crop = np.array((args.crop_width, args.crop_height))

        self.frame_timeout = 0 if args.timelapse else 1

    def update_fps(self):
        now = time.perf_counter()
        self.frame_time = max(now - self.frame_start_time, sys.float_info.epsilon)
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_text_with_background(self, frame, text, origin,
                                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0,
                                  color=(0, 0, 0), thickness=1, bgcolor=(255, 255, 255)):
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame,
                      tuple((origin + (0, baseline)).astype(int)),
                      tuple((origin + (text_size[0], -text_size[1])).astype(int)),
                      bgcolor, cv2.FILLED)
        cv2.putText(frame, text,
                    tuple(origin.astype(int)),
                    font, scale, color, thickness)
        return text_size, baseline

    def draw_person_detection_roi(self, frame, person_roi):
        # Draw person ROI border
        cv2.rectangle(frame,
                      tuple(person_roi.position_min), tuple(person_roi.position_max),
                      (0, 220, 0), 2)


    def draw_detection_roi(self, frame, roi, identity, age_gender, emotion):
        label = self.frame_processor \
            .face_identifier.get_identity_label(identity.id)

        # Draw face ROI border
        cv2.rectangle(frame,
                      tuple(roi.position), tuple(roi.position + roi.size),
                      (0, 220, 0), 2)

        # Draw identity label
        text_scale = 0.4
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("H1", font, text_scale, 1)
        line_height = np.array([0, text_size[0][1]])
        text = f"{label}, {age_gender.info}"
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text = f"{label.capitalize()} ({' %.2f%%' % (100.0 * (1 - identity.distance))}), {age_gender.info}"
        self.draw_text_with_background(frame, text,
                                       roi.position - line_height * 0.4,
                                       font, scale=text_scale)
        self.draw_text_with_background(frame, emotion.emotion_text,
                                       np.array((roi.position[0], roi.position[1] + roi.size[1])),
                                       font, scale=text_scale)

    def draw_detection_keypoints(self, frame, roi, landmarks):
        keypoints = [landmarks.left_eye,
                     landmarks.right_eye,
                     landmarks.nose_tip,
                     landmarks.left_lip_corner,
                     landmarks.right_lip_corner]

        for point in keypoints:
            center = roi.position + roi.size * point
            cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 2)

    def draw_detections(self, frame, detections):
        for roi, landmarks, identity, age_gender, emotion in zip(*detections[0]):
            self.draw_detection_roi(frame, roi, identity, age_gender, emotion)
            self.draw_detection_keypoints(frame, roi, landmarks)
        for person_roi in detections[1]:
            self.draw_person_detection_roi(frame, person_roi)

    def draw_status(self, frame, detections):
        origin = np.array([10, 10])
        color = (10, 160, 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text_size, _ = self.draw_text_with_background(frame,
                                                      "Frame time: %.3fs" % (self.frame_time),
                                                      origin, font, text_scale, color)
        self.draw_text_with_background(frame,
                                       "FPS: %.1f" % (self.fps),
                                       (origin + (0, text_size[1] * 1.5)), font, text_scale, color)

        log.debug('Frame: %s/%s, detections: %s, ' \
                  'frame time: %.3fs, fps: %.1f' % \
                     (self.frame_num, self.frame_count, len(detections[-1]), self.frame_time, self.fps))

        if self.print_perf_stats:
            log.info('Performance stats:')
            log.info(self.frame_processor.get_performance_stats())

    def display_interactive_window(self, frame):
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = "Press '%s' key to exit" % (self.BREAK_KEY_LABELS)
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, text,
                    tuple(origin.astype(int)), font, text_scale, color, thickness)
        cv2.imshow('Interactive Face recognition', frame)

    def should_stop_display(self):
        key = cv2.waitKey(self.frame_timeout) & 0xFF
        return key in self.BREAK_KEYS

    def process(self, input_stream, output_stream):
        self.input_stream = input_stream
        self.output_stream = output_stream
        while self.input_stream.isOpened():
            has_frame, frame = self.input_stream.read()
            if not has_frame:
                break

            if self.input_crop is not None:
                frame = Visualizer.center_crop(frame, self.input_crop)
            detections = self.frame_processor.process(frame)

            self.draw_detections(frame, detections)
            # self.draw_status(frame, detections)

            if self.output_stream:
                self.output_stream.write(frame)
            if self.display:
                self.display_interactive_window(frame)
                if self.should_stop_display():
                    break
            
            self.update_fps()
            self.frame_num += 1


    @staticmethod
    def center_crop(frame, crop_size):
        fh, fw, fc = frame.shape
        crop_size[0] = min(fw, crop_size[0])
        crop_size[1] = min(fh, crop_size[1])
        return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                     (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                     :]

    def run(self, args):
        input_stream = Visualizer.open_input_stream(args.input)
        if input_stream is None or not input_stream.isOpened():
            log.error("Cannot open input stream: %s" % args.input)
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        frame_size = (int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.crop_width and args.crop_height:
            crop_size = (args.crop_width, args.crop_height)
            frame_size = tuple(np.minimum(frame_size, crop_size))
        log.info("Input stream info: %d x %d @ %.2f FPS" % \
            (frame_size[0], frame_size[1], fps))
        if args.output == "":
            args.output = "output/{}.avi".format("cam" if args.input == "0" else args.input.split("/")[-1][:-4])
        output_stream = Visualizer.open_output_stream(args.output, fps, frame_size)

        self.process(input_stream, output_stream)

        # Release resources
        if output_stream:
            output_stream.release()
        if input_stream:
            input_stream.release()

        cv2.destroyAllWindows()

    @staticmethod
    def open_input_stream(path):
        log.info("Reading input data from '%s'" % (path))
        stream = path
        try:
            stream = int(path)
        except ValueError:
            pass
        return cv2.VideoCapture(stream)

    @staticmethod
    def open_output_stream(path, fps, frame_size):
        output_stream = None
        if path != "":
            if not path.endswith('.avi'):
                log.warning("Output file extension is not 'avi'. " \
                        "Some issues with output can occur, check logs.")
            log.info("Writing output to '%s'" % (path))
            output_stream = cv2.VideoWriter(path,
                                            cv2.VideoWriter.fourcc(*'MJPG'), fps, frame_size)
        return output_stream


def main():
    args = build_argparser().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    log.debug(str(args))

    visualizer = Visualizer(args)
    visualizer.run(args)


if __name__ == '__main__':
    main()
