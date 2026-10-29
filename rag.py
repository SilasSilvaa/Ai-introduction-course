from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader

load_dotenv()
directory = "documents"

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# Load Documents
pdf_loader = PyPDFDirectoryLoader(directory)
pdf_docs = pdf_loader.load()

text_loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
text_docs = text_loader.load()

docs = pdf_docs + text_docs

# Creating embeddings and vector
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)

messages = [
    SystemMessage(content="You are a helpful assistant that answers questions using the retrieved document context.")
]

while True:
    user_query = input("Enter your query (or 'exit' to quit): ")

    if user_query.lower() in {"exit", "quit", "sair"}:
        break

    retrieved_docs = vector_store.similarity_search(user_query, k=2)
    context = "\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"Use this document context to answer the user's question:\n{context}\n\nQuestion: {user_query}"
    messages.append(HumanMessage(content=prompt))

    response = llm.invoke(messages)

    print(response.text)
    messages.append(response)