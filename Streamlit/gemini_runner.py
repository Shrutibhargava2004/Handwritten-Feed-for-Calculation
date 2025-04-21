import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import base64
from google import genai

# Set up Gemini API
client = genai.Client(api_key="YOUR_API_KEY_HERE")  # <-- Replace with your actual key or load via env

def encode_image_to_base64(canvas):
    """Convert image (OpenCV) to base64 for Gemini API."""
    _, buffer = cv2.imencode('.png', canvas)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    return base64_img

def get_gemini_response(canvas):
    """Send canvas image to Gemini API and get response."""
    try:
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
        return response.text.strip()
    except Exception as e:
        return f"Gemini API Error: {e}"

def run_gemini_mode():
    """Launch webcam and use Gemini API for prediction."""
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    canvas = None
    prev_pos = None
    display_text = ""

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

            # Drawing with index finger
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

            # Erase with 4 fingers
            if fingers == [0, 1, 1, 1, 1]:
                eraser_pos = lmList[8][0:2]
                cv2.circle(canvas, tuple(eraser_pos), 20, (0, 0, 0), -1)

            # Trigger Gemini: Index + Pinky
            if fingers == [0, 1, 0, 0, 1]:
                display_text = get_gemini_response(canvas)
                print("Gemini Response:", display_text)

        # Overlay canvas + result
        combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        if display_text:
            cv2.putText(combined, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Canvas", canvas)
        cv2.imshow("Real-Time", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# When running this script directly
if __name__ == "__main__":
    run_gemini_mode()
