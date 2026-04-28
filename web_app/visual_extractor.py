import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')


def _download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face_landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model downloaded to {MODEL_PATH}")
    return MODEL_PATH


_MP_468_TO_68_MAPPING = [
    127, 356, 226, 168, 197, 195, 5, 4,
    219, 125, 19, 247, 238, 36, 39,
    68, 98, 152, 130, 136, 150, 176,
    172, 58, 93, 220, 45, 248, 193,
    415, 417, 370, 426, 436, 354, 356,
    7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161,
    246, 33, 7, 163, 144, 145, 153,
    154, 155, 133, 173, 157, 158, 159,
    160, 161, 246, 33,
]


def _extract_landmarks_68(face_landmarks, img_w, img_h):
    landmarks_68 = np.zeros((68, 2), dtype=np.float64)
    for i, mp_idx in enumerate(_MP_468_TO_68_MAPPING):
        lm = face_landmarks[mp_idx]
        landmarks_68[i, 0] = lm.x * img_w
        landmarks_68[i, 1] = lm.y * img_h
    return landmarks_68


def extract_visual_features(video_path, target_fps=30):
    _download_model()

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30

    frame_interval = max(1, int(round(video_fps / target_fps)))

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    features_list = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = landmarker.detect(mp_image)

        if result.face_landmarks and len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]
            landmarks_68 = _extract_landmarks_68(landmarks, w, h)
            features_list.append(landmarks_68.flatten())
        else:
            if features_list:
                features_list.append(features_list[-1].copy())
            else:
                features_list.append(np.zeros(136, dtype=np.float64))

        frame_idx += 1

    cap.release()
    landmarker.close()

    if not features_list:
        return np.zeros((1, 136), dtype=np.float64)

    features_array = np.array(features_list, dtype=np.float64)
    return features_array
