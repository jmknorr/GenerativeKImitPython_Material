from openai import OpenAI
from dotenv import load_dotenv
import os
from pprint import pprint

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
url_image = "https://i.redd.it/i5sh7ct49p121.jpg"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are the roles of the persons in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url_image,
                    },
                },
            ],
        }
    ],
)

pprint(response.choices[0].message.content)