import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=api_key
)

prompt = PromptTemplate.from_template(
    "You are a creative and helpful AI assistant. Based on the instruction below, generate high-quality and engaging text.\n\n"
    "Instruction: {instruction}\n\n"
    "Generated Text:"
)

chain: RunnableSequence = prompt | llm

instruction_input = "Write a  paragraph about the AI in Health."

result = chain.invoke({"instruction": instruction_input})

print("\nGenerated Text:\n")
print(result.content)
