import json
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter

# Initiate LLM model
llm = ChatOllama(model="llama3", temperature=0)

# Turn JSON data of questions and answers into a document for RAG
json_data = json.load(open('responses.json', 'r'))
splitter = RecursiveJsonSplitter(max_chunk_size=300)
docs = splitter.create_documents(texts=[json_data])

# Create retriever from document of questions and answers
vector_store = InMemoryVectorStore.from_documents(documents=docs, embedding=OllamaEmbeddings(model="llama3"))
retriever = vector_store.as_retriever()

# Define prompt with context from retriever to be used for answering questions
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "\n\n "
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Create RAG chain which uses retriever and QA chain to fetch answers from document
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# Function to invoke RAG chain from user query
def get_response(query: str) -> str:
    response = rag_chain.invoke({"input": query})
    return response['answer']