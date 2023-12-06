from typing import Any, List, Optional, Tuple

import mediapipe as mp
import numpy as np


class PoseRenderer:
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

    def draw(
        self,
        image: np.ndarray,
        pose_landmarks: Any,
        offset: Tuple[int, int] = (0, 0),
        scale: Tuple[float, float] = (1.0, 1.0)
    ) -> np.ndarray:
        for landmark in pose_landmarks.landmark:
            landmark.x *= scale[0]
            landmark.y *= scale[1]
            landmark.x += offset[0] / image.shape[1]
            landmark.y += offset[1] / image.shape[0]
        self.mp_drawing.draw_landmarks(
            image,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        return image


BODY_POINTS = [
    # LEFT_ARM_POINTS
    [
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0
    ],

    # RIGHT_ARM_POINTS
    [
        -1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0
    ],

    # BOTH_ARMS_POINTS
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0
    ],

    # ONLY_FACE
    [
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        0.0,
        0.0,
        -1.0,
        -1.0
    ]
]


class Pose:
    def __init__(self, model_complexity: int = 1) -> None:
        self.mp_pose = mp.solutions.pose
        self.predictor = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            static_image_mode=True)
        self.important_landmarks = [
            [(1.0, 0)],
            [(1.0, 11), (1.0, 13)],
            [(1.0, 12), (1.0, 14)],
            [(1.0, 13), (1.0, 15)],
            [(1.0, 14), (1.0, 16)],
            [(1.0, 15), (1.0, 17), (1.0, 19)],
            [(1.0, 16), (1.0, 18), (1.0, 20)],
            [(1.0, 15), (1.0, 17), (1.0, 19)],
            [(1.0, 16), (1.0, 18), (1.0, 20)],
            [(1.0, 23), (1.0, 25)],
            [(1.0, 24), (1.0, 26)],
            [(1.0, 25), (1.0, 27)],
            [(1.0, 26), (1.0, 28)],
            [(1.0, 27), (1.0, 29), (1.0, 31)],
            [(1.0, 28), (1.0, 30), (1.0, 32)],
            [(20.0, 11), (1.0, 12), (1.0, 23), (1.0, 24)],
            [(1.0, 11), (20.0, 12), (1.0, 23), (1.0, 24)],
            [(1.0, 11), (1.0, 12), (20.0, 23), (1.0, 24)],
            [(1.0, 11), (1.0, 12), (1.0, 23), (20.0, 24)]
        ]

    def predict_raw(self, image: np.ndarray) -> Any:
        return self.predictor.process(image)

    def combine_landmarks(
        self,
        image: np.ndarray,
        landmark_map: List[Tuple[float, int]],
        raw_landmarks: np.ndarray
    ) -> Tuple[float, float, float, float]:
        weight_sum = 0.0
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        sum_visibility = 0.0
        for weight, landmark_id in landmark_map:
            landmark = self.get_landmark(
                image, landmark_id, raw_landmarks)
            weight_sum += weight
            sum_x += weight * landmark[0]
            sum_y += weight * landmark[1]
            sum_z += weight * landmark[2]
            sum_visibility += weight * landmark[3]
        sum_x /= weight_sum
        sum_y /= weight_sum
        sum_z /= weight_sum
        sum_visibility /= weight_sum
        return (sum_x, sum_y, sum_z, sum_visibility)

    def predict(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raw_landmarks = self.predict_raw(image)
        if not raw_landmarks.pose_landmarks:
            return np.array([]), None
        important_landmarks = np.array([
            self.combine_landmarks(image, landmark_map,
                                   raw_landmarks.pose_landmarks)
            for landmark_map in self.important_landmarks
        ], dtype=float)

        return important_landmarks, raw_landmarks.pose_landmarks

    def get_landmark(
        self,
        image: np.ndarray,
        id: int,
        pose_landmarks: Any
    ) -> Tuple[float, float, float, float]:
        landmark = pose_landmarks.landmark[id]
        pos = (landmark.x * image.shape[1], landmark.y *
               image.shape[0], landmark.z, landmark.visibility)
        return pos

    def close(self) -> None:
        self.predictor.close()
