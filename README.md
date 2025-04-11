# 🏡 Multi-Agent Real Estate Assistant Chatbot (Text + Image)

This is a code-based chatbot that acts as a virtual assistant for real estate issues, using AI to answer both **property issue queries (via image + text)** and **tenancy FAQs (via text)**.

---

## 🚀 Features

- 🧠 **Agent 1: Property Issue Troubleshooter**
  - Accepts image uploads of property issues
  - Uses a BLIP image captioning model to describe the issue
  - GPT-4 gives troubleshooting suggestions

- 📚 **Agent 2: Tenancy Law FAQ Bot**
  - Handles legal queries about rental agreements, deposits, evictions, etc.
  - Provides location-specific advice (if location is given)

---

## 🛠 Tech Stack

| Layer       | Tool                                      |
|-------------|-------------------------------------------|
| Frontend    | Streamlit                                 |
| Backend     | LangChain + OpenAI GPT-4                  |
| Image Model | Salesforce BLIP (via HuggingFace)         |
| Env Mgmt    | `python-dotenv`                           |

---

## 🖥️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/real-estate-chatbot.git
cd real-estate-chatbot
```

### 2. Create `.env` file
```
OPENAI_API_KEY=your_openai_key_here
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🌐 Deployment

### ✅ Streamlit Cloud
1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your repo, set `OPENAI_API_KEY` in secrets
4. Deploy and share the link 🚀

---

## 🧪 Example Interactions

**🖼️ Image Upload (Agent 1):**
> User uploads image of wall + says: “What’s wrong with this wall?”
>
> → Bot responds: "There appears to be mold growth due to high humidity. Consider dehumidification."

**💬 Text-only Query (Agent 2):**
> User asks: "Can my landlord evict me without notice in London?"
>
> → Bot responds: "In the UK, landlords must give written notice unless in emergencies."

---

## 📄 License
MIT License.

---

## 🙋‍♂️ Author
Built by [Your Name]. Contributions welcome!

