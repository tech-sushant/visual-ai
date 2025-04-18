import base64
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from PIL import Image, ImageDraw
import cv2
import numpy as np
import io

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

def encode_image_to_base64(path, media_type):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded

def find_differences(image1_path, image2_path, output_path):
    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Ensure images are the same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply threshold to get regions with significant differences
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # thresholded image to fill in holes
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create output image
    output = img1.copy()
    
    # Draw rectangles around differences
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small differences
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 165, 255), 2)
    
    # Save output image
    cv2.imwrite(output_path, output)
    return output_path

image1_path = "image1.png"
image2_path = "image2.png"
image1_media_type = "image/png"
image2_media_type = "image/png"
image1_data = encode_image_to_base64(image1_path, image1_media_type)
image2_data = encode_image_to_base64(image2_path, image2_media_type)

# Generate the difference image
annotated_image_path = "difference_image.png"
find_differences(image1_path, image2_path, annotated_image_path)
annotated_image_data = encode_image_to_base64(annotated_image_path, "image/png")

# Get Claude's analysis
message = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    system="You are an UI/UX expert with great attention to detail.",
    messages=[
        {
            "role": "user", 
            "content": [
                {
                    "type": "text", 
                    "text": "Image 1:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image1_media_type,
                        "data": image1_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Image 2:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image2_media_type,
                        "data": image2_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Annotated Difference Image:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": annotated_image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "I've highlighted the differences between image 1 and image 2 with orange rectangles. Please provide a detailed analysis of these visual and functional changes and explain how they impact the user experience."
                }
            ],
        }
    ],
)

response = message.content[0].text

# Print the response and save it to a file
print("Response:", response)
with open("comparison.md", "w") as f:
    f.write(response)