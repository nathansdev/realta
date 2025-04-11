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

# Intent Classifier
classifier_prompt = PromptTemplate.from_template("""
Classify this user message into one of the following intents:
- issue (for physical property issues)
- faq (for rental agreements, tenancy laws, deposits, etc.)

Message: {text}

Respond with only one word: issue or faq
""")
intent_classifier = classifier_prompt | ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)

def classify_intent(text):
    result = intent_classifier.invoke({"text": text})
    return result.content.strip().lower()

def route_request(image_path=None, text=None, location=None):
    caption = generate_caption(image_path) if image_path else None

    if text:
        intent = classify_intent(text)

        if intent == "issue":
            result = agent1_chain.invoke({
                "image_caption": caption or "No image provided.",
                "user_text": text
            })
            response = result.content if hasattr(result, "content") else str(result)
            if not image_path:
                response += "\n\n📸 If you can, please upload an image of the issue for more accurate help."
            return response

        if intent == "faq":
            result = agent2_chain.invoke({
                "user_text": text,
                "location": location or "unspecified"
            })
            return result.content if hasattr(result, "content") else str(result)

    elif image_path:
        result = agent1_chain.invoke({
            "image_caption": caption,
            "user_text": ""
        })
        return result.content if hasattr(result, "content") else str(result)

    return "Please upload an image or enter a question."


# --- Streamlit Chat UI ---
st.set_page_config(page_title="🏡 Realtaa – Your Assistant for Property", layout="centered")

st.title("🏡 Realtaa – Your Assistant for Property")

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploader
user_image = st.file_uploader("Upload a property image (optional)")

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
user_input = st.chat_input("Ask a question or describe the issue...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if user_image:
                with open("temp.jpg", "wb") as f:
                    f.write(user_image.read())
                response = route_request("temp.jpg", user_input)
            else:
                response = route_request(text=user_input)
            st.markdown(response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})