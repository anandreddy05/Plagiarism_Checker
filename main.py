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
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import SecretStr
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API")
pinecone_api_key = os.getenv("PINECONE_API")

# Validate API keys
if not openai_api_key:
    raise ValueError("OPENAI_API environment variable is required")
if not pinecone_api_key:
    raise ValueError("PINECONE_API environment variable is required")

model = ChatOpenAI(api_key=SecretStr(openai_api_key), model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(api_key=SecretStr(openai_api_key), model="text-embedding-3-small")
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

# Vector Store setup with error handling
try:
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "plagiarism-detection"

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info(f"Created new Pinecone index: {index_name}")

    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    logger.info("Successfully connected to Pinecone vector store")

except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise

chain = prompt | model | parser

def extract_full_pdf_text(file_bytes: bytes) -> str:
    """
    Extract text from PDF bytes with error handling.
    
    Args:
        file_bytes: PDF file bytes
        
    Returns:
        Extracted text string
        
    Raises:
        HTTPException: If PDF extraction fails
    """
    try:
        pdf_stream = io.BytesIO(file_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        if doc.page_count == 0:
            doc.close()  # Don't forget to close the document
            raise HTTPException(status_code=400, detail="PDF file contains no pages")
        
        text = "\n".join([page.get_text("text") for page in doc])
        doc.close()
        
        # This check should come BEFORE the general exception handler
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF file contains no extractable text")
        
        return text
        
    except HTTPException:
        # Re-raise HTTP exceptions (including the one we just created above)
        raise
    except fitz.FileDataError as e:
        raise HTTPException(status_code=400, detail="Invalid PDF file format")
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF")


def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file.
    
    Args:
        file: Uploaded file
        
    Raises:
        HTTPException: If file validation fails
    """
    # Check file size (10MB limit)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size too large (max 10MB)")
    
    # Check file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Check content type
    if file.content_type and file.content_type not in ['application/pdf']:
        raise HTTPException(status_code=400, detail="Invalid content type. Expected application/pdf")

def parse_chain_output(output: str) -> tuple[str, str]:
    """
    Parse chain output into summary and skills.
    
    Args:
        output: Raw chain output
        
    Returns:
        Tuple of (summary, skills)
    """
    try:
        parts = output.split("SKILLS:")
        summary_text = parts[0].replace("SUMMARY:", "").strip()
        skills = parts[1].strip() if len(parts) > 1 else ""
        
        # Ensure we have some content
        if not summary_text:
            raise ValueError("No summary found in chain output")
        
        return summary_text, skills
        
    except Exception as e:
        logger.error(f"Failed to parse chain output: {e}")
        raise HTTPException(status_code=500, detail="Failed to process document content")

app = FastAPI(
    title="Plagiarism Detection API",
    description="API for detecting plagiarism in SRS documents",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Plagiarism detection service is running"}

@app.post("/check-plagiarism")
async def check_plagiarism(file: UploadFile = File(...)):
    """
    Check for plagiarism in uploaded PDF document.
    
    Args:
        file: PDF file to check
        
    Returns:
        JSON response with plagiarism detection results
    """
    try:
        # Validate file
        validate_file(file)
        
        # Read file
        file_bytes = await file.read()
        logger.info(f"Processing file: {file.filename}, size: {len(file_bytes)} bytes")
        
        # Extract text from PDF
        full_text = extract_full_pdf_text(file_bytes)
        
        # Process with AI chain
        try:
            output = chain.invoke({"text": full_text})
        except Exception as e:
            logger.error(f"Chain processing failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to process document with AI model")
        
        # Parse chain output
        summary_text, skills = parse_chain_output(output)
        
        # Create document for vector store
        new_doc = Document(
            page_content=summary_text,
            metadata={
                "source_file": file.filename,
                "skills": skills,
                "file_size": len(file_bytes)
            }
        )
        
        # Search for similar documents
        try:
            results = vector_store.similarity_search_with_score(summary_text, k=3)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to search for similar documents")
        
        # Analyze results
        plagiarism_detected = False
        matched_files = []
        max_score = 0
        threshold = 0.75
        
        for doc, score in results:
            if score > max_score:
                max_score = score
            if score >= threshold:
                plagiarism_detected = True
                source_file = doc.metadata.get("source_file")
                if source_file and source_file not in matched_files:
                    matched_files.append(source_file)
        
        # Add document to vector store if no plagiarism detected
        if not plagiarism_detected:
            try:
                vector_store.add_documents([new_doc])
                logger.info(f"Added document to vector store: {file.filename}")
                
                # Log index stats
                stats = index.describe_index_stats()
                logger.info(f"Pinecone index stats: {stats}")
                
            except Exception as e:
                logger.error(f"Failed to add document to vector store: {e}")
                # Don't raise exception here - plagiarism check was successful
        
        # Prepare response
        response_data = {
            "plagiarism_detected": plagiarism_detected,
            "max_score": round(max_score, 4),
            "matched_files": matched_files,
            "threshold": threshold,
            "document_added": not plagiarism_detected
        }
        
        logger.info(f"Plagiarism check completed for {file.filename}: {response_data}")
        return JSONResponse(response_data)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.exception_handler(413)
async def file_too_large_handler(request, exc):
    """Handle file too large errors."""
    return JSONResponse(
        status_code=413,
        content={"detail": "File too large"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
