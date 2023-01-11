import cv2
import mediapipe as mp
import math


class HandTracker:

    def __init__(self, mode=False, maxHands=1, detectionConfidence=0.8, trackingConfidence=0.8):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands
        self.mpDrawing = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionConfidence,
                                        min_tracking_confidence=self.trackingConfidence)
        self.distance = []

    def findLandmarks(self, frame):
        hands = self.hands
        # BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Set Flag False
        image.flags.writeable = False
        # Detections
        results = hands.process(image)
        # Set Flag True
        image.flags.writeable = True
        # RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def drawLandmarks(self, frame, results, wantAllLandmarks, landmarkOne, landmarkTwo, printDistance,
                      drawLinesBetweenLandmarks, clickAt):
        """
        0. WRIST                                         11. MIDDLE_FINGER_DIP
        1. THUMB_CMC                                     12. MIDDLE_FINGER_TIP
        2. THUMB_MCP                                     13. RING_FINGER_MCP.
        3. THUMB_IP                                      14. RING_FINGER_PIP
        4. THUMB_TIP                                     15. RING_FINGER_DIP
        5. INDEX_FINGER_MCP                              16. RING_FINGER_TIP
        6. INDEX_FINGER_PIP                              17. PINKY_MCP
        7. INDEX_FINGER_DIP                              18. PINKY_PIP
        8. INDEX_FINGER_TIP                              19. PINKY_DIP
        9. MIDDLE_FINGER_MCP                             20. PINKY_TIP
        10. MIDDLE_FINGER_PIP
        """
        mpDrawing = self.mpDrawing
        mpHands = self.mpHands
        for a, hand in enumerate(results.multi_hand_landmarks):
            if wantAllLandmarks:
                mpDrawing.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)
            # find the distance between two landmarks
            lmList = []
            for id, lm in enumerate(hand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([cx, cy])
            # ## uncomment below line to see the lmList array
            # print(lmList)
            x1, y1 = lmList[landmarkOne]
            x2, y2 = lmList[landmarkTwo]
            distance = int(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
            if printDistance:
                print(distance)

            # draw the distance line on the screen
            if drawLinesBetweenLandmarks:
                if distance > clickAt:
                    color = (255, 0, 255)
                else:
                    color = (100, 255, 100)
                cv2.circle(frame, (x1, y1), 10, color, cv2.FILLED)
                cv2.circle(frame, (x2, y2), 10, color, cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 7, color, cv2.FILLED)
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                self.distance = [distance, x, y]

    def getDistance(self):
        return self.distance