# %% Pakete
from groq import Groq
import base64
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# %% Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "our_scripts/test_img.jpeg"

# Getting the base64 string
base64_image = encode_image(image_path)

user_query = "Beschreibe das Bild in 3 Sätzen auf Deutsch."

# %% run code
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)

print(chat_completion.choices[0].message.content)