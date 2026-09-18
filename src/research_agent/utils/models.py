import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Lightweight model for OpenRouter with lower token cost and structured output support
model_name = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")

llm = ChatOpenAI(
    model=model_name,
    base_url=os.getenv("OPENAI_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    max_tokens=1000,
)