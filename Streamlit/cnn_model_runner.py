import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import time
import math

# Load your trained model
model = load_model("combined_model.h5")
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '!', 'sqrt']

# Set up webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
canvas = None
prev_pos = None
display_text = ""
symbols = []

def preprocess(symbol_img):
    """Preprocess symbol image for model prediction."""
    symbol_img = cv2.cvtColor(symbol_img, cv2.COLOR_BGR2GRAY)
    symbol_img = cv2.resize(symbol_img, (28, 28))
    symbol_img = symbol_img / 255.0
    symbol_img = symbol_img.reshape(1, 28, 28, 1)
    return symbol_img

def evaluate_expression(expr):
    try:
        expr = expr.replace('sqrt', 'math.sqrt')
        expr = expr.replace('^', '**')
        return str(eval(expr))
    except:
        return "Invalid"

def segment_and_predict(canvas):
    global display_text
    display_text = ""

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])  # Sort by x-coordinate

    prediction_string = ""
    for (x, y, w, h) in bounding_boxes:
        if w * h > 100:
            roi = canvas[y:y+h, x:x+w]
            processed = preprocess(roi)
            pred = model.predict(processed)
            predicted_class = class_names[np.argmax(pred)]
            prediction_string += predicted_class
            cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(canvas, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    display_text = prediction_string + " = " + evaluate_expression(prediction_string)

# Start real-time loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)

        # Draw on canvas with index finger
        if fingers == [0, 1, 0, 0, 0]:
            current_pos = lmList[8][0:2]
            if prev_pos is None:
                prev_pos = current_pos
            cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (255, 255, 255), 10)
            prev_pos = current_pos
        else:
            prev_pos = None

        # Clear canvas
        if fingers == [1, 1, 1, 1, 1]:
            canvas = np.zeros_like(img)
            display_text = ""

        # Erase
        if fingers == [0, 1, 1, 1, 1]:
            eraser_pos = lmList[8][0:2]
            cv2.circle(canvas, tuple(eraser_pos), 20, (0, 0, 0), -1)

        # Evaluate: Index + Pinky
        if fingers == [0, 1, 0, 0, 1]:
            segment_and_predict(canvas)

    # Display result
    combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    if display_text:
        cv2.putText(combined, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Canvas", canvas)
    cv2.imshow("Real-Time", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
