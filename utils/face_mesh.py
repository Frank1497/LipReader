import cv2 
import mediapipe as mp

class FaceMesh():

    def __init__(self, static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self. mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawSpec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.faceMesh = self.mp_face_mesh.FaceMesh(static_image_mode=self.static_image_mode, max_num_faces=self.max_num_faces, min_detection_confidence=self.min_detection_confidence,min_tracking_confidence=self.min_tracking_confidence)


    def get_landmarks(self, image):
        positions = []
        results = self.faceMesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks is not None:
            for landmarks in results.multi_face_landmarks:
                for id, lm in enumerate(landmarks.landmark):
                    h,w,c = image.shape
                    x = int(lm.x*w)
                    y = int(lm.y*h)
                    positions.append([id, x, y])
            return positions
        
        