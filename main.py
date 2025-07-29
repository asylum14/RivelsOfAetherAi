import cv2
import pytesseract
import numpy as np
import mss
from PIL import Image

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh

# Define the subregion for damage (example coords — tune this)
damage_box = {
    "top": 900,  # offset from monitor top
    "left": 755,  # offset from monitor left
    "width": 150,
    "height": 90
}
damage_box2 = {
    "top": 900,  # offset from monitor top
    "left": 955,  # offset from monitor left
    "width": 150,
    "height": 90
}
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"
lower_color = np.array([150, 71, 86])    # lower HSV bound
upper_color = np.array([173, 255, 222])   # upper HSV bound
prev_frame = None
contours = None
frameCount = 0
with mss.mss() as sct:
    
    while True:
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

         # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Create a binary mask where color matches
        mask = cv2.inRange(hsv, lower_color, upper_color)
       
         # Crop damage region
        damage_crop = frame[
            damage_box["top"] - monitor["top"] : damage_box["top"] - monitor["top"] + damage_box["height"],
            damage_box["left"] - monitor["left"] : damage_box["left"] - monitor["left"] + damage_box["width"]
        ]
        damage_crop2 = frame[
            damage_box2["top"] - monitor["top"] : damage_box2["top"] - monitor["top"] + damage_box2["height"],
            damage_box2["left"] - monitor["left"] : damage_box2["left"] - monitor["left"] + damage_box2["width"]
        ]
        # Preprocess and show
        
        #cv2.imshow("Damage Crop", damage_crop)
        cv2.imshow("Damage Crop", damage_crop2)
        if frameCount%20==0:
            p1Damage = pytesseract.image_to_string(damage_crop, config='--psm 7 digits')
            p2Damage = pytesseract.image_to_string(damage_crop2, config='--psm 7 digits')

            print(f"Player 1 Damage:{p1Damage.strip()}, Player 2 Damage:{p2Damage.strip()}")

        # Find contours (blobs of color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Choose the largest blob
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

            largest =sorted_contours[0]
            secondLargest = sorted_contours[1]

            (x, y, w, h) = cv2.boundingRect(largest)
            center = (x + w // 2, y + h // 2)
            (x2, y2, w2, h2) = cv2.boundingRect(secondLargest)
            SecondCenter = (x2 + w2 // 2, y2 + h2 // 2)
            middleOfChar = (((x+w//2)+(x2+w2//2))//2, ((y+h//2)+(y2+h2//2))//2)

            # Draw the position on screen
            cv2.circle(frame, middleOfChar, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{middleOfChar}", ((x+x2)//2, ((y+y2)//2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1,cv2.LINE_AA)
        prev_frame = gray.copy()
        cv2.imshow("Game View", frame)
        frameCount+=1
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break


cv2.destroyAllWindows()