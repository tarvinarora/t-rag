import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
logging.getLogger("pypdf").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

loader = PyPDFDirectoryLoader("t-rag/pdfs")
docs = loader.load()
# print(docs[0].page_content)

# chunking strategy - RecursiveCharacterTextSplitter => even though structured documents, 
# they are PDFs and markdown structrued chunking startegies will not be able to isolate structure 
# in this case so we stick to a simple chunking strategy and will evaluate later. 

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, #average paragraph size in documents
    chunk_overlap = 200, 
    separators=["\n\n", "\n",". ", " ", ""],
)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

for c in chunks[:3]:
    print(repr(c.page_content[:200]))
    print("source:", c.metadata["source"], "page:", c.metadata["page"])
    print("---")

#chunk health checks
lengths = [len(c.page_content) for c in chunks]
print(f"chunks: {len(chunks)}")
print(f"min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)//len(lengths)}")

# how many are suspiciously tiny? (usually junk: page numbers, headers)
tiny = [c for c in chunks if len(c.page_content) < 100]
print(f"tiny chunks: {len(tiny)}")

#Local embeddings + Chroma store
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

#Load instead of rebuild on later runs

if os.path.exists("./chroma_db"):
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function = embeddings,
    )
    print("Loaded existing DB")
else:
    vector_db = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    persist_directory = "./chroma_db",
    )
    print("New Vctor DB built")


# Retriever + Prompt + LLM + loop
retriever = vector_db.as_retriever(search_kwargs={"k":3})
template = """You are a helpful assistant. Answer the question using ONLY the context below. If the answer
is not in the context, say "I don't know based on the provided documents." 

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template = template,
    input_variables = {"context", "question"}
)

llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash")

#one text block for {context} NOT a list of objects
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

while True:
    question = input("\nYour question (or 'quit'): ")
    if question.lower() == "quit":
        break

    docs = retriever.invoke(question) #retrieve the relevant chunks

    print("\n--- Retrieved from ---")
    for d in docs:
        print(f"{d.metadata['source']} p.{d.metadata['page']}") #Eval = bad chunks? or bad generation?

    context = format_docs(docs) #format the prompt ready for LLM
    final_prompt = prompt.format(context = context, question = question)

    answer = llm.invoke(final_prompt) #ask the LLM

    content = answer.content
    if isinstance(content, list):
        text = "".join(block["text"] for block in content if block.get("type") == "text")
    else:
        text = content

    print("\nAnswer:", text)

