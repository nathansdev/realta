import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import torch
import os
import tempfile
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

# Agent 1: Issue Detection with memory
agent1_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
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
agent1_llm = ChatOpenAI(temperature=0.5, model="gpt-4", openai_api_key=openai_api_key)
agent1_chain = agent1_prompt | agent1_llm

# Agent 2: Tenancy FAQ with memory
agent2_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent2_prompt = PromptTemplate.from_template("""
You're a tenancy expert. Answer the question below accurately and clearly.
If the user's location is provided, give local guidance.

Location: {location}
Question: {user_text}

Answer:
""")
agent2_llm = ChatOpenAI(temperature=0.5, model="gpt-4", openai_api_key=openai_api_key)
agent2_chain = agent2_prompt | agent2_llm

# Intent Classifier
classifier_prompt = PromptTemplate.from_template("""
Classify this user message into one of the following intents:
- issue (for physical property issues like cracks, leaks, damages, maintenance problems)
- faq (for rental agreements, tenancy laws, deposits, legal rights, contracts, rental terms)
- gratitude (for thank you messages)

Examples:
- "there's a crack in my wall" -> issue
- "what are my rights as a tenant" -> faq
- "what is the tenancy law in my area" -> faq
- "there's mold in the bathroom" -> issue
- "how much notice should my landlord give" -> faq
- "thanks for the help" -> gratitude

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

        # Check if the question might be better suited for the other agent
        if st.session_state.selected_agent == "faq" and intent == "issue":
            st.markdown("It seems like you're asking about a property issue. This would be better handled by our Issue Detection Agent.")
            st.markdown("Would you like to switch to the Issue Detection Agent?")
            if st.button("Switch to Issue Detection Agent", use_container_width=True):
                st.session_state.messages = []
                st.session_state.image_path = None
                st.session_state.user_image = None
                st.session_state.selected_agent = "issue"
                st.rerun()
            return ""

        if st.session_state.selected_agent == "issue" and intent == "faq":
            st.markdown("It seems like you're asking about tenancy or legal matters. This would be better handled by our Tenancy FAQ Agent.")
            st.markdown("Would you like to switch to the Tenancy FAQ Agent?")
            if st.button("Switch to Tenancy FAQ Agent", use_container_width=True):
                st.session_state.messages = []
                st.session_state.image_path = None
                st.session_state.user_image = None
                st.session_state.selected_agent = "faq"
                st.rerun()
            return ""

        if intent == "gratitude":
            return "You're welcome! Feel free to ask if you have any other questions about your property."

        if intent not in ["issue", "faq"]:
            return "I'm not sure how to help yet. Could you tell me if it's about a property issue or a tenancy question?"

        if intent == "issue":
            if not image_path:
                return "Can you upload a photo of the issue so I can help you better?"

            st.write("BLIP Image Description:", caption)
            # Update memory with current context
            agent1_memory.save_context(
                {"input": f"Image: {caption}\nUser: {text}"},
                {"output": ""}
            )
            result = agent1_chain.invoke({
                "image_caption": caption or "No image provided.",
                "user_text": text
            })
            response = result.content if hasattr(result, "content") else str(result)
            return response

        if intent == "faq":
            # Update memory with current context
            agent2_memory.save_context(
                {"input": f"Location: {location}\nQuestion: {text}"},
                {"output": ""}
            )
            result = agent2_chain.invoke({
                "user_text": text,
                "location": location or "unspecified"
            })
            return result.content if hasattr(result, "content") else str(result)

    elif image_path:
        st.write("BLIP Image Description:", caption)
        # Update memory with current context
        agent1_memory.save_context(
            {"input": f"Image: {caption}\nUser: The user uploaded an image but didn't type anything."},
            {"output": ""}
        )
        result = agent1_chain.invoke({
            "image_caption": caption,
            "user_text": "The user uploaded an image but didn't type anything."
        })
        return "🖼️ Based on the image alone:\n\n" + (result.content if hasattr(result, "content") else str(result))

    return "Please upload an image or enter a question."

# --- Streamlit Chat UI ---
st.set_page_config(page_title="🏡 Realtaa – Your Assistant for Property", layout="centered")

# Add JavaScript for agent switching
st.markdown("""
<script>
function switchAgent(agent) {
    window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        value: agent
    }, '*');
}
</script>
""", unsafe_allow_html=True)

st.title("🏡 Realtaa – Your Assistant for Property")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_image" not in st.session_state:
    st.session_state.user_image = None
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = None

# Initial agent selection
if st.session_state.selected_agent is None:
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h3>👋 Welcome to Realtaa!</h3>
        <p>Please select an agent to get started:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Issue Detection Agent", use_container_width=True):
            st.session_state.selected_agent = "issue"
            st.rerun()
    with col2:
        if st.button("📚 Tenancy FAQ Agent", use_container_width=True):
            st.session_state.selected_agent = "faq"
            st.rerun()
else:
    # Show agent selection buttons at the top
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "🔍 Issue Detection Agent", 
            use_container_width=True,
            type="primary" if st.session_state.selected_agent == "issue" else "secondary"
        ):
            # Clear messages when switching agents
            st.session_state.messages = []
            st.session_state.selected_agent = "issue"
            st.session_state.image_path = None  # Clear any existing image
            st.session_state.user_image = None
            st.rerun()
    with col2:
        if st.button(
            "📚 Tenancy FAQ Agent", 
            use_container_width=True,
            type="primary" if st.session_state.selected_agent == "faq" else "secondary"
        ):
            # Clear messages when switching agents
            st.session_state.messages = []
            st.session_state.selected_agent = "faq"
            st.session_state.image_path = None  # Clear any existing image
            st.session_state.user_image = None
            st.rerun()

    # Add some custom CSS for better button styling
    st.markdown("""
    <style>
    div[data-testid="stButton"] button {
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #4CAF50;
        border-color: #4CAF50;
    }
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #f0f0f0;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display current agent
    st.markdown(f"**Current Agent:** {'🔍 Issue Detection' if st.session_state.selected_agent == 'issue' else '📚 Tenancy FAQ'}")

    # Add Clear Chat button
    if st.session_state.messages:  # Only show clear button if there are messages
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.image_path = None
            st.session_state.user_image = None
            st.rerun()

    # Location input
    user_location = st.text_input("📍 Where are you located? (Optional)")

    # File uploader (only show for issue detection agent)
    if st.session_state.selected_agent == "issue":
        uploaded_image = st.file_uploader("Upload a property image (optional)", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            with st.spinner("Processing image..."):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                temp_file.write(uploaded_image.getvalue())
                st.session_state.user_image = uploaded_image
                st.session_state.image_path = temp_file.name
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

    # Chat input
    user_input = st.chat_input("Ask a question or describe the issue...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your request..."):
                if st.session_state.selected_agent == "issue":
                    if st.session_state.image_path:
                        response = route_request(st.session_state.image_path, user_input, user_location)
                    else:
                        response = route_request(text=user_input, location=user_location)
                else:  # faq agent
                    response = route_request(text=user_input, location=user_location)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# Mobile-friendly CSS
st.markdown("""
<style>
@media screen and (max-width: 768px) {
  h1, h2, h3 { font-size: 1.2rem !important; }
  .stTextInput input { font-size: 16px !important; }
}
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
