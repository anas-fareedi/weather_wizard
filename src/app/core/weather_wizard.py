try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"MediaPipe import error: {e}")

import numpy as np
import cv2
import random

from utils.gesture_detector import GestureDetector
from core.particle_system import Particle

class WeatherWizard:
    def __init__(self):
        # Try to initialize MediaPipe for hand detection
        if MEDIAPIPE_AVAILABLE:
            try:
                # Download hand landmarker model if needed
                import urllib.request
                import os
                model_path = "hand_landmarker.task"
                if not os.path.exists(model_path):
                    print("Downloading hand detection model...")
                    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                    urllib.request.urlretrieve(url, model_path)
                    print("Model downloaded!")
                
                # Create HandLandmarker for new API
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=1,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                self.use_gestures = True
                print("MediaPipe hand detection initialized successfully!")
            except Exception as e:
                print(f"MediaPipe initialization failed: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to keyboard-only controls")
                self.use_gestures = False
        else:
            self.use_gestures = False

        print(" Attempt to open camera .") 
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("failed to  open webcam, please check your camera connection.")
        print(" Camera opened successfully .")

        success, image = self.cap.read()
        self.height, self.width = image.shape[:2]
        self.particle_system = Particle(self.width, self.height)
        
        # Initialize gesture detector only if MediaPipe is working
        if self.use_gestures:
            self.gesture_detector = GestureDetector()

        self.particles = []
        self.max_particles = 100
        self.current_effect = None

        self.RAIN_COLOR = (200, 255, 255)
        self.SNOW_COLOR = (255, 255, 255)
        self.LIGHTNING_COLOR = (255, 255, 100)
        
        print("Controls:")
        if self.use_gestures:
            print("Use hand gestures:")
            print("- 1 finger up: Lightning")
            print("- 2 fingers up: Snow") 
            print("- 3 fingers up: Rain")
        print("Keyboard controls:")
        print("- Press '1' for Lightning")
        print("- Press '2' for Snow") 
        print("- Press '3' for Rain")
        print("- Press 'ESC' to exit")
    
    def run(self):
        """Main loop to run the Weather Wizard application."""
        
        rain_active = False
        snow_active = False
        lightning_active = False
        
        while True:
            success, image = self.cap.read()
            if not success:
                print("fail to read from camera.")
                break

            image = cv2.flip(image, 1)
            
            # Process hand gestures if MediaPipe is available
            gesture_detected = None
            x_pos = self.width // 2
            fingers_count = 0
            
            if self.use_gestures:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # Detect hands using new API
                detection_result = self.detector.detect(mp_image)
                
                if detection_result.hand_landmarks:
                    for hand_landmarks in detection_result.hand_landmarks:
                        
                        gesture_detected = self.gesture_detector.detect_gestures(hand_landmarks)
                        
                        index_tip = hand_landmarks[8]
                        x_pos = int(index_tip.x * self.width)
                        
                        fingers_count = 0
                        if hand_landmarks[8].y < hand_landmarks[6].y:  
                            fingers_count += 1
                        if hand_landmarks[12].y < hand_landmarks[10].y:  
                            fingers_count += 1
                        if hand_landmarks[16].y < hand_landmarks[14].y:  
                            fingers_count += 1
                        if hand_landmarks[20].y < hand_landmarks[18].y: 
                            fingers_count += 1
                            
                        cv2.putText(image, f"Fingers: {fingers_count}", (10, 110), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            key = cv2.waitKey(1) & 0xFF
            
            rain_active = False
            snow_active = False
            lightning_active = False
            particle_x = self.width // 2
            
            if gesture_detected == "LIGHTNING":
                print("Activating LIGHTNING from gesture")
                lightning_active = True
                particle_x = x_pos
            elif gesture_detected == "SNOW":
                print("Activating SNOW from gesture")
                snow_active = True
                particle_x = x_pos
            elif gesture_detected == "RAIN":
                print("Activating RAIN from gesture")
                rain_active = True
                particle_x = x_pos
            
            # Keyboard fallback (only if no gesture detected)
            if not gesture_detected:
                if key == ord('1'):
                    print("Activating LIGHTNING from keyboard")
                    lightning_active = True
                elif key == ord('2'):
                    print("Activating SNOW from keyboard")
                    snow_active = True
                elif key == ord('3'):
                    print("Activating RAIN from keyboard")
                    rain_active = True
            
            if key == 27:  
                break
            if lightning_active:
                self.current_effect = "LIGHTNING"
            elif snow_active:
                self.current_effect = "SNOW"
            elif rain_active:
                self.current_effect = "RAIN"
            else:
                self.current_effect = None
            
            if rain_active and len(self.particles) < self.max_particles:
                # Create multiple rain particles across the screen width
                for _ in range(5):  
                    if len(self.particles) < self.max_particles:
                        x_pos_rain = random.randint(0, self.width)
                        self.particles.append(
                            self.particle_system.create_particle("RAIN", x_pos_rain))
            
            elif snow_active and len(self.particles) < self.max_particles:
                # Create multiple snow particles across the screen width
                for _ in range(3): 
                    if len(self.particles) < self.max_particles:
                        x_pos_snow = random.randint(0, self.width)
                        self.particles.append(
                            self.particle_system.create_particle("SNOW", x_pos_snow))
            
            elif lightning_active and len(self.particles) < 5: 
                x_pos_lightning = particle_x if particle_x else random.randint(self.width // 4, 3 * self.width // 4)
                self.particles.append(
                    self.particle_system.create_particle("LIGHTNING", x_pos_lightning))
            
            if self.current_effect:
                cv2.putText(image, f"Effect: {self.current_effect} (Active)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image, "No Effect Active", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if self.use_gestures and gesture_detected:
                cv2.putText(image, f"Gesture: {gesture_detected}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            control_text = "Gestures: 1/2/3 fingers | Keys: 1/2/3 | ESC=Exit" if self.use_gestures else "Keys: 1=Lightning, 2=Snow, 3=Rain, ESC=Exit"
            cv2.putText(image, control_text, 
                       (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.update_particles(image)
            cv2.imshow('Weather Wizard', image)
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def update_particles(self, image):
        """ update and draw all particles"""
        new_particles = []
        overlay = np.zeros_like(image)

        for particle in self.particles:
            if particle['type'] == 'rain':
                particle['y'] += particle['speed']
                if particle['y'] < self.height:
                    cv2.line(
                        overlay,
                        (int(particle['x']), int(particle['y'])),
                        (int(particle['x']), int(particle['y']) - 30),
                        self.RAIN_COLOR,
                        3
                    )
                    new_particles.append(particle)
                    
            elif particle['type'] == 'snow':
                particle['y'] += particle['speed']
                particle['x'] += particle['drift']
                if 0 < particle['y'] < self.height and 0 < particle['x'] < self.width:
                    cv2.circle(
                        overlay,
                        (int(particle['x']), int(particle['y'])),
                        5,
                        self.SNOW_COLOR,
                        -1
                    )
                    new_particles.append(particle)
            
            elif particle['type'] == 'lightning':
                particle['lifetime'] -= 0.2
                if particle['lifetime'] > 0:
                    
                    for i in range(len(particle['segments']) - 1):
                        start = particle['segments'][i]
                        end = particle['segments'][i + 1]

                        start_x = max(0, min(self.width, int(start[0])))
                        start_y = max(0, min(self.height, int(start[1])))
                        end_x = max(0, min(self.width, int(end[0])))
                        end_y = max(0, min(self.height, int(end[1])))

                        cv2.line(
                            overlay,
                            (start_x, start_y),
                            (end_x, end_y),
                            self.LIGHTNING_COLOR,
                            4
                        )
                        if random.random() < 0.3:
                            branch_end_x = end_x + random.randint(-40, 40)
                            branch_end_y = end_y + random.randint(20, 60)
                            branch_end_x = max(0, min(self.width, branch_end_x))
                            branch_end_y = max(0, min(self.height, branch_end_y))
                            cv2.line(
                                overlay,
                                (end_x, end_y),
                                (branch_end_x, branch_end_y),
                                self.LIGHTNING_COLOR,
                                2
                            )
                    if particle['lifetime'] > 4:
                        flash_overlay = np.full_like(image, 255)
                        overlay = cv2.addWeighted(overlay, 1, flash_overlay, particle['flash_intensity'], 0)
                    
                    new_particles.append(particle)
        result = cv2.addWeighted(image, 1, overlay, 0.7, 0)
        image[:] = result[:]
        self.particles = new_particles
        