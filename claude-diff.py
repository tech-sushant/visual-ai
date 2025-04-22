import base64
import os
import json
import re
import cv2
from dotenv import load_dotenv
from anthropic import Anthropic

# Load API key
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

# Resize image to 951px width
def resize_for_claude(image_path, target_width=951):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    scale_ratio = target_width / w
    new_h = int(h * scale_ratio)
    resized = cv2.resize(img, (target_width, new_h))
    return img, resized, scale_ratio

# Convert OpenCV image to base64 string
def encode_cv2_image_to_base64(cv2_img):
    _, buffer = cv2.imencode(".png", cv2_img)
    return base64.b64encode(buffer).decode("utf-8")

# Paths to your two images
image1_path = "image1.png"
image2_path = "image2.png"

# Load and resize
original1, resized1, scale1 = resize_for_claude(image1_path)
original2, resized2, scale2 = resize_for_claude(image2_path)

# Encode resized images
image1_data = encode_cv2_image_to_base64(resized1)
image2_data = encode_cv2_image_to_base64(resized2)

# Claude API call with improved prompt
message = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    system=(
        "You are a professional visual comparison AI. Your ONLY job is to compare two UI screenshots "
        "and return differences in a structured format. IMPORTANT: Always return the full bounding box "
        "around a complete logical UI component where a change occurs — not just the pixel or text that changed. "
        "Err on the side of slightly larger bounding boxes to include any padding/margins for readability. "
        "Examples include: entire button (including icon/text/background), whole heading block, card or container."
    ),
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image1_data,
                    },
                },
                {"type": "text", "text": "Image 2"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image2_data,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Compare these UI screenshots and return ONLY a JSON array. "
                        "Each item must be an object with a 'label' and 'coordinates'. "
                        "Coordinates = [x1, y1, x2, y2] on the resized image. "
                        "Highlight the FULL component where a change happens (e.g. full button, entire heading, complete icon + label). "
                        "Be generous — include margin/padding to avoid tight boxes. Do NOT respond with any explanation, just the JSON array."
                    )
                }
            ],
        }
    ],
)

# Extract JSON array from response
response_text = message.content[0].text.strip()
match = re.search(r"\[\s*{.*?}\s*\]", response_text, re.DOTALL)
if not match:
    print("No valid JSON array found in the response.")
    exit()

json_data = match.group(0)
try:
    annotations = json.loads(json_data)
except json.JSONDecodeError as e:
    print("Failed to parse JSON:", e)
    exit()

# Draw bounding boxes on original2
for ann in annotations:
    label = ann.get("label", "Change")
    coords = ann.get("coordinates", [])

    # Scale coordinates to original image size
    x1 = int(coords[0] / scale2)
    y1 = int(coords[1] / scale2)
    x2 = int(coords[2] / scale2)
    y2 = int(coords[3] / scale2)

    # Draw box and label
    cv2.rectangle(original2, (x1, y1), (x2, y2), (0, 165, 255), 2)
    cv2.putText(original2, label, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1, cv2.LINE_AA)

# Save result
output_path = "difference_image.png"
cv2.imwrite(output_path, original2)
print(f"Saved difference image as {output_path}")