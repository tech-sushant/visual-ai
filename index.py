import base64
import os
from dotenv import load_dotenv
load_dotenv()
from anthropic import Anthropic

api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

def encode_image_to_base64(path, media_type):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded

image1_path = "image1.png"
image1_media_type = "image/png"
image1_data = encode_image_to_base64(image1_path, image1_media_type)

image2_path = "image2.png"
image2_media_type = "image/png"
image2_data = encode_image_to_base64(image2_path, image2_media_type)


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
                    "text":
                    "You are given two images. Identify the visual and functional changes between them. Explain how these changes can impact the user experience (UX)"
                }
            ],
        }
    ],
)

response = message.content[0].text

# Print the response and save it to a file
print("Response:", response)
with open("comparison.txt", "w") as f:
    f.write(response)