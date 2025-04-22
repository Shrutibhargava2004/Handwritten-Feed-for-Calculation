import cvzone
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

# Load model
model = load_model("combined_model.h5")

# Class labels
class_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "add", "subtract", "multiply", "divide", "!", "(", ")", "[", "]", "{", "}", "pi", "sqrt"]

# Symbol mappings
symbol_map = {
    "add": "+", "subtract": "-", "multiply": "*", "divide": "/",
    "!": "!", "(": "(", ")": ")", "[": "[", "]": "]",
    "{": "{", "}": "}", "pi": "math.pi", "sqrt": "math.sqrt"
}

# Init webcam and set resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

prev_pos = None
canvas = None
predicted_operator = None
expression = ""
result = ""
symbol_removed = False

# --- Helper Functions ---

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None 
    if fingers == [0, 1, 0, 0, 0]:  # Only index up
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (255, 255, 255), 10)
    return current_pos, canvas

def preprocess_for_prediction(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    y_indices, x_indices = np.where(binary > 0)

    if len(y_indices) > 0 and len(x_indices) > 0:
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        cropped = binary[y_min:y_max+1, x_min:x_max+1]
    else:
        return None

    h, w = cropped.shape
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = 20
        new_h = int(20 / aspect_ratio)
    else:
        new_h = 20
        new_w = int(20 * aspect_ratio)

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    final_img = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    final_img = final_img / 255.0
    return final_img.reshape(1, 28, 28, 1)

def evaluate_expression(expression):
    expression = expression.replace("!", "math.factorial")
    try:
        result = eval(expression, {"__builtins__": None}, {"math": math, "pi": math.pi, "sqrt": math.sqrt})
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# --- Main Loop ---

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info

        if fingers == [1, 1, 1, 1, 1]:  # Clear canvas
            canvas = np.zeros_like(img)
            prev_pos = None
            predicted_operator = None
            expression = ""
            result = ""
            print("Canvas & Expression Cleared")

        if fingers == [0, 1, 0, 0, 1]:  # Predict and add symbol
            processed_img = preprocess_for_prediction(canvas)
            if processed_img is not None:
                prediction = model.predict(processed_img)
                predicted_class = np.argmax(prediction)
                predicted_label = class_labels[predicted_class]
                predicted_operator = predicted_label
                symbol = symbol_map.get(predicted_label, predicted_label)
                expression += symbol
                print("Added:", symbol)
            canvas = np.zeros_like(img)
            prev_pos = None

        if fingers == [0, 1, 1, 1, 1] and expression != "":  # Evaluate
            result = evaluate_expression(expression)
            print("Evaluated:", result)

        if fingers == [0, 0, 0, 0, 1] and expression != "":  # Remove last
            if not symbol_removed:
                removed_symbol = expression[-1]
                expression = expression[:-1]
                predicted_operator = None
                symbol_removed = True
                print(f"Removed last symbol: {removed_symbol}")
        else:
            symbol_removed = False

        prev_pos, canvas = draw(info, prev_pos, canvas)

    combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

    # Smaller Font Display
    if predicted_operator:
        cv2.putText(combined, f'Predicted: {predicted_operator}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if expression:
        cv2.putText(combined, f'Expression: {expression}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if result:
        cv2.putText(combined, f'Result: {result}', (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Canvas", canvas)
    cv2.imshow("Air Math", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
