# coding: utf-8
# import the necessary packages
from imutils import face_utils
import numpy as np
import dlib
import cv2
import os
import matplotlib.pyplot as plt

class DetectFace:
    def __init__(self, image):
        # initialize dlib's face detector (HOG-based)
        # and then create the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.abspath("shape_predictor_68_face_landmarks.dat"))
        
        # face detection part
        self.img = cv2.imread(image)
        if self.img is None:
            print("이미지를 로드할 수 없습니다.")
            return
        
        # init face parts
        self.right_eyebrow = []
        self.left_eyebrow = []
        self.right_eye = []
        self.left_eye = []
        self.left_cheek = []
        self.right_cheek = []

        # detect the face parts and set the variables
        self.detect_face_part()

    # return type : np.array
    def detect_face_part(self):
        face_parts = []
        
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            print("얼굴을 찾을 수 없습니다.")
            return

        # 첫 번째 얼굴만 처리 (여러 얼굴이 감지된 경우)
        rect = faces[0]

        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        idx = 0
        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if i < len(shape) and j <= len(shape):  # 인덱스가 안전한지 확인
                face_parts.append(shape[i:j])

        if len(face_parts) < 5:
            return

        face_parts = face_parts[1:5]  # 얼굴 부위만 추출

        # set the variables
        if len(face_parts) >= 4:
            self.right_eyebrow = self.extract_face_part(face_parts[0])
        
            self.left_eyebrow = self.extract_face_part(face_parts[1])
        
            self.right_eye = self.extract_face_part(face_parts[2])
        
            self.left_eye = self.extract_face_part(face_parts[3])
        
            # Cheeks are detected by relative position to the face landmarks
            self.left_cheek = self.img[shape[29][1]:shape[33][1], shape[4][0]:shape[48][0]]
        
            self.right_cheek = self.img[shape[29][1]:shape[33][1], shape[54][0]:shape[12][0]]

    # parameter example : self.right_eye
    # return type : image
    def extract_face_part(self, face_part_points):
        (x, y, w, h) = cv2.boundingRect(face_part_points)
        crop = self.img[y:y+h, x:x+w]
        adj_points = np.array([np.array([p[0]-x, p[1]-y]) for p in face_part_points])

        # Create an mask
        mask = np.zeros((crop.shape[0], crop.shape[1]))
        cv2.fillConvexPoly(mask, adj_points, 1)
        mask = mask.astype(bool)
        crop[np.logical_not(mask)] = [255, 0, 0]

        return crop
