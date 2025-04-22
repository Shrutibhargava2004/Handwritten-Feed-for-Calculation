import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import base64
from google import genai
import time

# Initialize Gemini Client
client = genai.Client(api_key="AIzaSyCGA0SzK_-hjMsDZGh7vGJHOpO9RwTEtuE")  # Replace with your actual key
# Convert image to base64
def encode_image_to_base64(canvas):
    _, buffer = cv2.imencode('.png', canvas)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    return base64_img

# Get response from Gemini
def get_gemini_response(canvas):
    try:
        print("[INFO] Encoding canvas and calling Gemini API...")
        base64_img = encode_image_to_base64(canvas)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64_img
                            }
                        },
                        {
                            "text": "Solve this handwritten math expression and return only the final result like '12' or '3.14'."
                        }
                    ]
                }
            ]
        )
        print("[INFO] Gemini API responded.")
        return response.text.strip()
    except Exception as e:
        print("[ERROR] Gemini API failed:", e)
        return f"Gemini API Error: {e}"

# Webcam and canvas setup
def run_gemini_calculator():
    # print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check if it’s in use or not connected.")
        return

    detector = HandDetector(maxHands=1)
    canvas = None
    prev_pos = None
    display_text = ""

    # print("[INFO] Webcam started successfully.")
    time.sleep(1)

    while True:
        success, img = cap.read()
        if not success:
            print("[WARNING] Failed to read frame from webcam.")
            continue

        img = cv2.flip(img, 1)

        if canvas is None:
            canvas = np.zeros_like(img)

        hands, img = detector.findHands(img, draw=True, flipType=True)
        if hands:
            print("[INFO] Hand detected.")
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = detector.fingersUp(hand)
            # print("[DEBUG] Fingers up:", fingers)

            # Drawing
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
                # print("[ACTION] Clearing canvas.")
                canvas = np.zeros_like(img)
                display_text = ""

            # Erase
            if fingers == [0, 1, 1, 1, 1]:
                # print("[ACTION] Erasing.")
                eraser_pos = lmList[8][0:2]
                cv2.circle(canvas, tuple(eraser_pos), 20, (0, 0, 0), -1)

            # Trigger Gemini
            if fingers == [0, 1, 0, 0, 1]:
                # print("[ACTION] Sending /canvas to Gemini API.")
                display_text = get_gemini_response(canvas)
                # print("[RESULT]", display_text)

        combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        if display_text:
            cv2.putText(combined, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Canvas", canvas)
        cv2.imshow("Real-Time", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # print("[INFO] Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera and windows released.")

# Entry point
if __name__ == "__main__":
    run_gemini_calculator()