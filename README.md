# RAG based Q&A Chatbot
This is a demo of how to build a Retrieval Augemented Generation based Chatbot to answer questions based on a JSON document of questions and answers.

## Installation
First, clone the repository using

```bash
git clone https://github.com/prabhatreddy/RAG-Chatbot-Demo.git
```

This demo uses [Poetry](https://python-poetry.org/docs/) to manage dependencies. If you do not have Poetry installed, follow the instructions provided in their documentation to install it.

To initialize all dependencies, run
```bash
poetry install
```

## LLM Provider

This demo uses [Ollama](https://github.com/ollama/ollama) as the LLM Provider. 
Be sure to pull the `llama3` model using

```bash
ollama pull llama3
```

It is relatively easy to change model providers to the provider of your choice. Do note that you will need to provide your own API keys. For example, to use OpenAI instead of Ollama make the following changes in `agent.py`:

```diff
import json
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
- from langchain_ollama import ChatOllama, OllamaEmbeddings
+ from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter
```

```diff
# Initiate LLM model
- llm = ChatOllama(model="llama3", temperature=0)
+ os.get_environ("OPENAI_API_KEY")
+ llm = ChatOpenAI()
```

```diff
# Create retriever from document of questions and answers
- vector_store = InMemoryVectorStore.from_documents(documents=docs, embedding=OllamaEmbeddings(model="llama3"))
+ vector_store = InMemoryVectorStore.from_documents(documents=docs, embedding=OpenAIEmbeddings())
retriever = vector_store.as_retriever()
```

(Do not forget that you will need to install `langchain-openai` or the langchain extension for the model provider of your choice)

## Usage
For this demo, you will need to create a JSON document of questions and answers called 'responses.json' in the following format:

```JSON
{
    "questions": [
        {
            "question": "<Question 1>",
            "answer" : "<Answer 1>"
        },
        {
            "question": "<Question 2>",
            "answer" : "<Answer 2>"
        },
    ]
}
```
You can use the template file `template.json` to help do this.

To run the demo

```bash
streamlit run app.py
```