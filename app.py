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
analyze the issue and provide specific details.

Image says: {image_caption}
User says: {user_text}

If the user is asking about counting or identifying specific elements (like cracks, stains, etc.):
1. First, acknowledge that you're working from an image description
2. If the image description provides specific counts or details, share those
3. If the image description is general, explain what you can determine from it
4. Suggest how to get more accurate information (e.g., closer inspection, professional assessment)

For other types of questions:
- Identify the issue
- Provide analysis
- Suggest fixes

Respond with clear, specific information based on what you can determine from the image description.
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
- gratitude (for thank you messages)

Message: {text}

Respond with only one word: issue, faq, or gratitude
""")
intent_classifier = classifier_prompt | ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)

def classify_intent(text):
    result = intent_classifier.invoke({"text": text})
    return result.content.strip().lower()

def route_request(image_path=None, text=None, location=None):
    caption = generate_caption(image_path) if image_path else None

    if text:
        intent = classify_intent(text)

        if intent == "gratitude":
            return "You're welcome! Feel free to ask if you have any other questions about your property."

        if intent == "issue":
            # For testing: Show the raw BLIP caption
            st.write("BLIP Image Description:", caption)
            
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
        # For testing: Show the raw BLIP caption
        st.write("BLIP Image Description:", caption)
        
        result = agent1_chain.invoke({
            "image_caption": caption,
            "user_text": ""
        })
        return result.content if hasattr(result, "content") else str(result)

    return "Please upload an image or enter a question."


# --- Streamlit Chat UI ---
st.set_page_config(page_title="🏡 Realtaa – Your Assistant for Property", layout="centered")

st.title("🏡 Realtaa – Your Assistant for Property")

# Session state for messages and image
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_image" not in st.session_state:
    st.session_state.user_image = None
if "image_path" not in st.session_state:
    st.session_state.image_path = None

# File uploader
uploaded_image = st.file_uploader("Upload a property image (optional)", type=["jpg", "jpeg", "png"])
if uploaded_image:
    with st.spinner("Processing image..."):
        # Save the uploaded file to a temporary location
        temp_path = f"temp_{uploaded_image.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_image.getvalue())
        st.session_state.user_image = uploaded_image
        st.session_state.image_path = temp_path
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        st.success("Image processed successfully!")

# Display message history
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h3>👋 Welcome to Realtaa!</h3>
        <p>I'm your property assistant. You can:</p>
        <ul style='list-style-type: none; padding: 0;'>
            <li>📸 Upload an image of a property issue</li>
            <li>💬 Ask questions about property maintenance</li>
            <li>📚 Get information about tenancy laws</li>
        </ul>
        <p>How can I help you today?</p>
    </div>
    """, unsafe_allow_html=True)
else:
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
        with st.spinner("Analyzing your request..."):
            if st.session_state.image_path:
                response = route_request(st.session_state.image_path, user_input)
            else:
                response = route_request(text=user_input)
            st.markdown(response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add custom CSS for better loading indicators
st.markdown("""
    <style>
    .stSpinner > div {
        background-color: #4CAF50;
        border-radius: 10px;
        padding: 10px;
        color: white;
    }
    .stSuccess {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Clean up temporary files when session ends
if st.session_state.image_path and os.path.exists(st.session_state.image_path):
    os.remove(st.session_state.image_path)
    st.session_state.image_path = None
