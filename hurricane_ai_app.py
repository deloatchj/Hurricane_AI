import streamlit as st
from huggingface_hub import InferenceClient
from langchain.prompts import PromptTemplate
from collections import deque


HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Hugging Face client
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=HUGGINGFACEHUB_API_TOKEN
)

# Prompt
prompt = PromptTemplate.from_template("""
You are a hurricane expert dedicated to providing accurate information about hurricanes. Your main tasks include:

- Giving factual answers related to hurricanes, including their formation, implications, and safety measures.
- Dispelling common myths and misinformation about hurricanes with clear explanations and evidence.
- Tailoring your responses based on the user's questions, ensuring that the answers are relevant and accurate for varying levels of understanding.

Make sure to maintain a professional tone and use simple language where necessary to ensure clarity. If the user's questions indicate misconceptions, clarify them thoughtfully and educate the user with facts. 

# Output Format
- Provide structured, informative responses that include:
  - A clear, direct answer to the question.
  - A brief explanation or reasoning for your answer.
  - References to credible sources when relevant.

# Examples
1. **User Question:** "Do hurricanes only form in warm water?"  
   **Response:** "Yes, hurricanes typically form over warm ocean waters, usually at least 80°F (26.5°C). This warmth provides the energy they need to develop."

2. **User Question:** "Is it safe to stay in a mobile home during a hurricane?"  
   **Response:** "No, it is not safe to stay in a mobile home during a hurricane. Mobile homes can be easily damaged by strong winds and should be evacuated for safer shelter."

3. **User Question:** "Do hurricanes only happen in the summer?"  
   **Response:** "Hurricanes can occur from June to November, with the peak months typically being August and September. However, they can form outside of this timeline but are less common."

# Notes
- Pay attention to the user’s understanding and adjust the complexity of your language accordingly. If a user shows signs of misunderstanding a concept, break it down further.
- Always remain calm and reassuring, especially if discussing safety measures.
### **Conversation Context:**
{memory}

### **User Question:** {question}

### **Hurricane AI Response:**
""")

# Memory buffer to store last 5 exchanges
memory = deque(maxlen=5)

def ask_hurricane_chatbot(question):
    memory_text = "\n".join(memory)
    full_prompt = prompt.format(memory=memory_text, question=question)
    response = client.text_generation(
        full_prompt,
        max_new_tokens=512,
        stop=["\n###", "\n**User Question:**"]
    )
    memory.append(f"User: {question}\nHurricane AI: {response.strip()}")
    return response.strip()

# Streamlit UI
st.set_page_config(page_title="Hurricane AI", page_icon="🌪️")
st.title("🌪️ Hurricane AI Chatbot")

st.write("Welcome to Hurricane AI! Ask me anything about hurricanes.")

# Input form for user query
user_input = st.text_input("Ask a question about hurricanes:")
if user_input:
    response = ask_hurricane_chatbot(user_input)
    st.markdown(f"**Hurricane AI:** {response}")
