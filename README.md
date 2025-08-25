# SRS Plagiarism Detection System

A FastAPI-based plagiarism detection system specifically designed for Software Requirements Specification (SRS) documents.

The system uses vector embeddings and semantic similarity to detect potential plagiarism in uploaded PDF documents.

## 🚀 Features

- **PDF Document Processing**: Extracts and processes text from PDF files using PyMuPDF
- **Intelligent Content Analysis**: Focuses on key SRS sections (Purpose, Product Scope, Product Perspective, Product Functions, System Features)
- **Vector-Based Similarity Detection**: Uses OpenAI embeddings and Pinecone vector database for semantic similarity matching
- **Automated Summarization**: Generates concise summaries and extracts technical skills/technologies using GPT-4o-mini
- **REST API Interface**: Easy-to-use FastAPI endpoints with comprehensive error handling
- **Similarity Scoring**: Provides detailed similarity scores and matched file references
- **File Validation**: Supports PDF files up to 10MB with proper format validation

## 📋 Prerequisites

- Python 3.12+ (recommended) or 3.8+
- OpenAI API key
- Pinecone API key
- PDF documents in SRS format

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anandreddy05/Plagiarism_Checker.git
   cd Plagiarism_Checker
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
- **Spec**: Serverless for cost-effective scaling

## 🚀 Usage

### Starting the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

The API will be available at `http://localhost:5000`

### Docker Deployment

```bash
# Build the image
docker build -t srs-plagiarism-detector .

# Run the container
docker run -p 5000:5000 srs-plagiarism-detector
```

### API Endpoints

#### Health Check
**GET** `/health`

Check if the service is running.

**Response**:
```json
{
  "status": "healthy",
  "message": "Plagiarism detection service is running"
}
```

#### Check Plagiarism
**POST** `/check-plagiarism`

Upload a PDF file to check for plagiarism against existing documents in the database.

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: PDF file (form field: `file`)
- File Requirements:
  - Format: PDF only
  - Size: Maximum 10MB
  - Content: Must contain extractable text

**Response**:
```json
{
  "plagiarism_detected": false,
  "max_score": 0.65,
  "matched_files": [],
  "threshold": 0.75,
  "document_added": true
}
```

**Response Fields**:
- `plagiarism_detected`: Boolean indicating if plagiarism was detected (similarity ≥ 0.75)
- `max_score`: Highest similarity score found (0.0 to 1.0)
- `matched_files`: List of filenames with high similarity scores (≥ 0.75)
- `threshold`: Current plagiarism detection threshold (0.75)
- `document_added`: Whether the document was added to the database (only if no plagiarism detected)

## 🧠 How It Works

1. **Document Processing**: 
   - Validates uploaded PDF file (format, size, content)
   - Extracts text from PDF using PyMuPDF (fitz)
   - Filters content to focus on key SRS sections

2. **Content Analysis**:
   - Uses GPT-4o-mini to generate summaries and extract technical skills
   - Creates semantic embeddings using OpenAI's text-embedding-3-small model
   - Separates summary content from technical skills list

3. **Similarity Detection**:
   - Performs vector similarity search in Pinecone database
   - Compares against top 3 most similar existing documents
   - Applies 0.75 threshold for plagiarism detection

4. **Database Management**:
   - Stores new documents only if no plagiarism is detected
   - Maintains metadata including source filename, extracted skills, and file size
   - Provides index statistics for monitoring

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

Other sections (UI details, hardware specs, legal, glossary) are filtered out to focus on core content.

## 📁 Project Structure

```
Plagiarism_Checker/
├── main.py                # Main FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── test_app.py          # Comprehensive test suite
├── .env                 # Environment variables (create this)
└── README.md           # This file
```

## 🛡️ Error Handling

The system handles various error scenarios:

- **File Validation Errors** (400):
  - Invalid PDF files or corrupted format
  - Files exceeding 10MB limit
  - Non-PDF file extensions
  - Empty or missing filenames

- **Processing Errors** (500):
  - Network connectivity issues with external APIs
  - AI model processing failures
  - Vector database connection issues

- **Content Errors** (400):
  - PDFs with no extractable text
  - Empty PDF files

## 🔒 Security Considerations

- API keys are loaded from environment variables only
- No sensitive data is logged in application logs
- File uploads are processed in memory without persistent local storage
- Vector database connections use secure API authentication
- Input validation prevents malicious file uploads
- File size limits prevent resource exhaustion

## 📈 Performance

- **Embedding Model**: OpenAI text-embedding-3-small (1536 dimensions)
- **LLM Model**: GPT-4o-mini for cost-effective summarization
- **Vector Search**: Cosine similarity with k=3 for efficiency
- **Database**: Pinecone serverless for scalable vector operations
- **File Processing**: In-memory PDF processing with 10MB limit
- **Error Recovery**: Graceful handling of API failures and timeouts

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio reportlab

# Run all tests
pytest test_app.py -v

# Run with coverage
pytest test_app.py --cov=main

# Run integration tests (requires API keys)
pytest test_app.py -m integration
```

The test suite covers:
- PDF text extraction and validation
- Plagiarism detection logic
- Error handling scenarios
- API endpoint functionality
- Edge cases and boundary conditions

## 🐳 Docker Support

The application includes Docker support for easy deployment:

```dockerfile
# Base image: Python 3.12-slim
# Port: 5000
# Command: uvicorn main:app --host 0.0.0.0 --port 5000
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest test_app.py`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **"OPENAI_API environment variable is required"**
   - Ensure your `.env` file contains valid API keys
   - Check that the `.env` file is in the project root directory

2. **"Failed to initialize Pinecone"**
   - Verify your Pinecone API key is correct
   - Check your internet connection
   - Ensure Pinecone service is available

3. **"Invalid PDF file format"**
   - Ensure the uploaded file is a valid PDF
   - Try re-saving the PDF from another application
   - Check if the PDF contains extractable text

4. **"File too large"**
   - Reduce PDF file size to under 10MB
   - Use PDF compression tools if needed

### API Rate Limits

- OpenAI: Monitor your usage and billing
- Pinecone: Check your plan limits and quota

For additional support, please create an issue on GitHub with detailed error information.
