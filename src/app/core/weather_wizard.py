import mediapipe as mp
import numpy as np
import cv2
import random

from app.utils.gesture_detector import GestureDetector
from app.core.particle_system import Particle

class WeatherWizard:
    def __init__(self):
        # Try to initialize MediaPipe for hand detection
        try:
            # MediaPipe hand detection (legacy API that still works)
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_gestures = True
            print("MediaPipe hand detection initialized successfully!")
        except Exception as e:
            print(f"MediaPipe initialization failed: {e}")
            print("Falling back to keyboard-only controls")
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
            if self.use_gestures:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_image)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Detect gesture
                        gesture_detected = self.gesture_detector.detect_gestures(hand_landmarks)
                        
                        # Get finger tip position for particle placement
                        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        x_pos = int(index_tip.x * self.width)
            
            # Handle keyboard input for effects
            key = cv2.waitKey(1) & 0xFF
            
            # Process gesture or keyboard input
            effect_triggered = None
            particle_x = None
            
            if gesture_detected == "LIGHTNING" or key == ord('1'):
                effect_triggered = "LIGHTNING"
                lightning_active = not lightning_active
                rain_active = False
                snow_active = False
                particle_x = x_pos if self.use_gestures and gesture_detected else self.width // 2
                        
            elif gesture_detected == "SNOW" or key == ord('2'):
                effect_triggered = "SNOW"
                snow_active = not snow_active
                rain_active = False
                lightning_active = False
                particle_x = x_pos if self.use_gestures and gesture_detected else self.width // 2
                        
            elif gesture_detected == "RAIN" or key == ord('3'):
                effect_triggered = "RAIN"
                rain_active = not rain_active
                snow_active = False
                lightning_active = False
                particle_x = x_pos if self.use_gestures and gesture_detected else self.width // 2
            
            elif key == 27:  # ESC key
                break
            
            # Update current effect
            if effect_triggered:
                self.current_effect = effect_triggered if (rain_active or snow_active or lightning_active) else None
            
            # Continuously generate particles for active effects
            if rain_active and len(self.particles) < self.max_particles:
                # Create multiple rain particles across the screen width
                for _ in range(5):  # Create 5 rain drops per frame
                    if len(self.particles) < self.max_particles:
                        x_pos_rain = random.randint(0, self.width)
                        self.particles.append(
                            self.particle_system.create_particle("RAIN", x_pos_rain))
            
            elif snow_active and len(self.particles) < self.max_particles:
                # Create multiple snow particles across the screen width
                for _ in range(3):  # Create 3 snowflakes per frame
                    if len(self.particles) < self.max_particles:
                        x_pos_snow = random.randint(0, self.width)
                        self.particles.append(
                            self.particle_system.create_particle("SNOW", x_pos_snow))
            
            elif lightning_active and len(self.particles) < 5:  # Limit lightning
                x_pos_lightning = particle_x if particle_x else random.randint(self.width // 4, 3 * self.width // 4)
                self.particles.append(
                    self.particle_system.create_particle("LIGHTNING", x_pos_lightning))
            
            # Add text overlays
            if self.current_effect:
                cv2.putText(image, f"Effect: {self.current_effect} (Active)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image, "No Effect Active", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show current gesture if detected
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


                                    