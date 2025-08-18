from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from pinecone import Pinecone,ServerlessSpec
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import fitz
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

load_dotenv()

openai_api_key = os.getenv("OPENAI_API")
pinecone_api_key = os.getenv("PINECONE_API")

model = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini",temperature=0)
embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
parser = StrOutputParser()

prompt = PromptTemplate(
    template="""
You are a summarization assistant for student SRS documents.

Only consider the following **key sections** from the text:
- 1.1 Purpose
- 1.4 Product Scope
- 2.1 Product Perspective
- 2.2 Product Functions
- 4.1 System Features

Ignore sections like UI details, hardware specs, legal, glossary, etc.

**Task**:
1. Create a concise and semantically rich summary (5–7 bullet points) **without including explicit lists of skills or technologies**.
2. Extract the skills/technologies separately as a comma-separated list.

Summary should cover:
- Project title
- Goal or problem being solved
- Main features or modules
- Any unique or innovative aspects

Skills should be:
- Programming languages
- Frameworks
- Libraries
- Tools
- Cloud platforms
- Databases

Format your answer as:
SUMMARY:
<summary text>

SKILLS:
<comma-separated skills list>

SRS Text:
{text}
""",
    input_variables=["text"]
)

# Vector Store
pc = Pinecone(api_key=pinecone_api_key)

from pinecone import ServerlessSpec

index_name = "plagiarism-detection" 

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

chain = prompt | model | parser

directory = "./SRS"


def extract_full_pdf_text(file_bytes: bytes):
    pdf_stream = io.BytesIO(file_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])


app = FastAPI()

@app.post("/check-plagiarism")
async def check_plagiarism(file: UploadFile = File(...)):
    file_bytes = await file.read()
    full_text = extract_full_pdf_text(file_bytes=file_bytes)
    output = chain.invoke({"text":full_text})
    
    parts = output.split("SKILLS:")
    summary_text = parts[0].replace("SUMMARY:","").strip()
    skills = parts[1].strip() if len(parts) > 1 else ""
        
    new_doc = Document(
        page_content= summary_text,
        metadata = {
            "source_file": file.filename,
            "skills": skills
        }
    )
    
    results = vector_store.similarity_search_with_score(summary_text,k=3)
    plagiarism_detected = False
    matched_files = []
    max_score = 0
    
    for doc,score in results:
        if score > max_score:
            max_score = score
        if score  >= 0.75:
            plagiarism_detected = True
            matched_files.append(doc.metadata.get("source_file"))
    
    if not plagiarism_detected:
        vector_store.add_documents([new_doc])
        print("ADDED to Pinecone:", new_doc.metadata["source_file"])
        print(index.describe_index_stats())

    
    return JSONResponse({
        "plagiarism_detected": plagiarism_detected,
        "max_score": max_score,
        "matched_files": matched_files
    })
