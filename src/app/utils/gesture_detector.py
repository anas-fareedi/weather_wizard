try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class GestureDetector:
    def __init__(self):
        pass

    def detect_gestures(self, hand_landmarks):
        """ Analyze hand landmarks to determine weather effect """
        
        if not MEDIAPIPE_AVAILABLE:
            return None
        
        # hand_landmarks is now a list of NormalizedLandmark objects
        # Landmark indices for finger tips and PIPs
        # Index finger (8 = tip, 6 = pip)
        index_tip_y = hand_landmarks[8].y
        index_pip_y = hand_landmarks[6].y
        
        # Middle finger (12 = tip, 10 = pip) 
        middle_tip_y = hand_landmarks[12].y
        middle_pip_y = hand_landmarks[10].y
        
        # Ring finger (16 = tip, 14 = pip)
        ring_tip_y = hand_landmarks[16].y
        ring_pip_y = hand_landmarks[14].y
        
        # Little finger (20 = tip, 18 = pip)
        little_tip_y = hand_landmarks[20].y
        little_pip_y = hand_landmarks[18].y
        
        # Count extended fingers (excluding thumb for clearer gestures)
        fingers_up = 0
            
        # Check fingers - tip should be above pip (lower y value means higher on screen)
        if index_tip_y < index_pip_y:
            fingers_up += 1
        if middle_tip_y < middle_pip_y:
            fingers_up += 1
        if ring_tip_y < ring_pip_y:
            fingers_up += 1
        if little_tip_y < little_pip_y:
            fingers_up += 1
        
        # Map finger count to effects
        print(f"Detected {fingers_up} fingers up")  # Debug output
        if fingers_up == 1:
            print("Returning LIGHTNING")
            return "LIGHTNING"
        elif fingers_up == 2:
            print("Returning SNOW")
            return "SNOW"
        elif fingers_up == 3:
            print("Returning RAIN")
            return "RAIN"

        return None