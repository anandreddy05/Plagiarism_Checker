# SRS Plagiarism Detection System

A FastAPI-based plagiarism detection system specifically designed for Software Requirements Specification (SRS) documents. The system uses vector embeddings and semantic similarity to detect potential plagiarism in uploaded PDF documents.

## 🚀 Features

- **PDF Document Processing**: Extracts and processes text from PDF files
- **Intelligent Content Analysis**: Focuses on key SRS sections (Purpose, Product Scope, Product Perspective, Product Functions, System Features)
- **Vector-Based Similarity Detection**: Uses OpenAI embeddings and Pinecone vector database for semantic similarity matching
- **Automated Summarization**: Generates concise summaries and extracts technical skills/technologies
- **REST API Interface**: Easy-to-use FastAPI endpoints for integration
- **Similarity Scoring**: Provides detailed similarity scores and matched file references

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key
- PDF documents in SRS format

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd plagiarism-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API=your_openai_api_key_here
   PINECONE_API=your_pinecone_api_key_here
   ```

4. **Create SRS directory** (optional)
   ```bash
   mkdir SRS
   ```

## 🔧 Configuration

### API Keys Setup

1. **OpenAI API Key**:
   - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create a new API key
   - Add to your `.env` file

2. **Pinecone API Key**:
   - Visit [Pinecone Console](https://app.pinecone.io/)
   - Create a new project or use existing
   - Generate API key and add to `.env` file

### Vector Database

The system automatically creates a Pinecone index named `plagiarism-detection` with:
- **Dimension**: 1536 (OpenAI text-embedding-3-small)
- **Metric**: Cosine similarity
- **Cloud**: AWS (us-east-1 region)

## 🚀 Usage

### Starting the Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Check Plagiarism
**POST** `/check-plagiarism`

Upload a PDF file to check for plagiarism against existing documents in the database.

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: PDF file (form field: `file`)

**Response**:
```json
{
  "plagiarism_detected": false,
  "max_score": 0.65,
  "matched_files": []
}
```

**Response Fields**:
- `plagiarism_detected`: Boolean indicating if plagiarism was detected (similarity ≥ 0.75)
- `max_score`: Highest similarity score found (0.0 to 1.0)
- `matched_files`: List of filenames with high similarity scores

## 🧠 How It Works

1. **Document Processing**: 
   - Extracts text from uploaded PDF using PyMuPDF
   - Filters content to focus on key SRS sections

2. **Content Analysis**:
   - Uses GPT-4o-mini to generate summaries and extract technical skills
   - Creates semantic embeddings using OpenAI's text-embedding-3-small

3. **Similarity Detection**:
   - Performs vector similarity search in Pinecone database
   - Compares against top 3 most similar existing documents
   - Applies 0.75 threshold for plagiarism detection

4. **Database Management**:
   - Stores new documents only if no plagiarism is detected
   - Maintains metadata including source filename and extracted skills

## 📊 Similarity Threshold

- **Similarity Score ≥ 0.75**: Flagged as potential plagiarism
- **Similarity Score < 0.75**: Document is considered original and added to database

## 🔍 Supported Document Sections

The system specifically analyzes these SRS sections:
- **1.1 Purpose**: Project objectives and goals
- **1.4 Product Scope**: Scope and boundaries
- **2.1 Product Perspective**: System context and relationships
- **2.2 Product Functions**: Core functionalities
- **4.1 System Features**: Detailed feature descriptions

## 📁 Project Structure

```
plagiarism-detection/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── SRS/                  # Optional directory for SRS documents
└── README.md            # This file
```

## 🛡️ Error Handling

The system handles:
- Invalid PDF files
- Network connectivity issues with external APIs
- Malformed SRS documents
- API rate limiting

## 🔒 Security Considerations

- API keys are loaded from environment variables
- No sensitive data is logged
- File uploads are processed in memory without persistent storage
- Vector database connections use secure API authentication

## 📈 Performance

- **Embedding Model**: OpenAI text-embedding-3-small (1536 dimensions)
- **LLM Model**: GPT-4o-mini for cost-effective summarization
- **Vector Search**: Cosine similarity with k=3 for efficiency
- **Database**: Pinecone serverless for scalable vector operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request
