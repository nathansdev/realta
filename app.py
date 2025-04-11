### real_estate_chatbot/app.py

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Agent 1: Issue Detection
agent1_prompt = PromptTemplate.from_template("""
You're a property issue expert. Given the following image description and user context,
identify the issue and suggest fixes.

Image says: {image_caption}
User says: {user_text}

Respond with analysis and suggestions.
""")
agent1_chain = agent1_prompt | ChatOpenAI(temperature=0.5, model="gpt-4", openai_api_key=openai_api_key)

# Agent 2: Tenancy FAQ
agent2_prompt = PromptTemplate.from_template("""
You're a tenancy expert. Answer the question below accurately and clearly.
If the user's location is provided, give local guidance.

Location: {location}
Question: {user_text}

Answer:
""")
agent2_chain = agent2_prompt | ChatOpenAI(temperature=0.5, model="gpt-4", openai_api_key=openai_api_key)

def route_request(image_path=None, text=None, location=None):
    if image_path:
        caption = generate_caption(image_path)
        return agent1_chain.invoke({
            "image_caption": caption,
            "user_text": text or ""
        })
    elif text:
        return agent2_chain.invoke({
            "user_text": text,
            "location": location or "unspecified"
        })
    else:
        return "Please upload an image or enter a question."

# Streamlit UI
st.title("🏡 Real Estate Assistant Chatbot")

user_text = st.text_input("Ask your question:")
user_image = st.file_uploader("Or upload an image:")
location = st.text_input("Your location (optional):")

if st.button("Submit"):
    if user_image:
        with open("temp.jpg", "wb") as f:
            f.write(user_image.read())
        response = route_request("temp.jpg", user_text)
    else:
        response = route_request(text=user_text, location=location)
    st.markdown("### 🤖 Chatbot Response:")
    st.write(response)
