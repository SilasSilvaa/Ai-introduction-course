from pathlib import Path
from pypdf import PdfReader

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage

# Load APIKey
load_dotenv()


def load_documents(directory="documents"):
    documents = []

    # Documents and System prompt
    for file_path in Path(directory).rglob("*"):
        if file_path.suffix in {".txt", ".pdf", ".md"}:
            if file_path.suffix == ".pdf":
                reader = PdfReader(file_path)
                content = "\n".join(page.extract_text() for page in reader.pages)
            else:
                content = file_path.read_text()
            documents.append({"file_path": str(file_path), "content": content})

    return documents


def create_context(document_list):
    context = ""

    for doc in document_list:
        context = context + f"{doc['file_path']}:\n{doc['content']}\n ----------- \n"
    return context


documents = load_documents()
context = create_context(documents)

system_prompt = f"""
You are a helpful assistant than answers question about these documents {context} 
"""

# Set up message for LLM
messages = [SystemMessage(content=system_prompt)]

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

while True:
    # Get user query
    user_query = input("Enter your query: ")

    # Append the user query to messages
    messages.append(HumanMessage(content=user_query))

    # Invoke the LLM
    response = llm.invoke(messages)

    # Display the response and add to messages
    print(response.text)
    messages.append(response)
