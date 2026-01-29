import mediapipe as mp

class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands

    def detect_gestures(self, hand_landmarks):
        """ Analyze hand landmarks to determine weather effect """
        
        # Get landmark points using MediaPipe's landmark structure
        landmarks = hand_landmarks.landmark
        
        # Landmark indices for finger tips and PIPs
        # Index finger
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        
        # Middle finger  
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        
        # Ring finger
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y
        ring_pip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
        
        # Thumb (check x-coordinate for thumb)
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP].x
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP].x

        # Count extended fingers
        fingers_up = 0
        
        # Check thumb (horizontal movement)
        if abs(thumb_tip - thumb_ip) > 0.03:  # Thumb extended
            fingers_up += 1
            
        # Check other fingers (vertical movement)
        if index_tip < index_pip:  # Index finger extended
            fingers_up += 1
        if middle_tip < middle_pip:  # Middle finger extended
            fingers_up += 1
        if ring_tip < ring_pip:  # Ring finger extended
            fingers_up += 1
        
        # Map finger count to effects
        if fingers_up == 1:
            return "LIGHTNING"
        elif fingers_up == 2:
            return "SNOW"
        elif fingers_up >= 3:
            return "RAIN"

        return None