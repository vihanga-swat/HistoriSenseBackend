# 🏛️ HistoriSense

AI-Powered War Testimony Analysis System  
Transforming historical documents into structured insights through advanced NLP

Badges: Python 3.8+, Flask 2.0+, MongoDB, OpenAI, Google Gemini

---

## Table of Contents
- Overview
- Features
- System Architecture
- Getting Started
- Configuration
- Running the Server
- API Reference
- Data & AI Components
- Database Schema
- File Processing & Limits
- Security & Best Practices
- Troubleshooting
- Roadmap
- Contributing
- License
- Acknowledgements
- Contact

---

## Overview

HistoriSense Backend is a Flask-based REST API for analyzing historical war testimonies and similar narratives. It processes uploaded documents, performs AI-driven extraction and classification, and stores structured insights for retrieval and research.

Project info:
- Developer: Vihanga Palihakkara
- Role: Associate Software Engineer
- Email: vihangawork@gmail.com
- Type: Individual project (FYP/Personal R&D)
- Goal: Convert historical documents into searchable, structured data using modern NLP/LLM tooling.

---

## Features

- Authentication & Users
  - JWT authentication
  - Individual and Museum roles
  - Password hashing with bcrypt
  - Email and password validation

- AI Analysis
  - Emotional analysis (top 5 emotions with distribution)
  - Geographical entity extraction (cities, countries, regions)
  - Topic classification (key themes with prominence scores)
  - Writer/author info extraction (biographical cues)
  - People mentioned (names, roles, regions)

- Document Processing
  - Supported: PDF, DOCX, TXT
  - Safe upload, text extraction, size limits (16MB)
  - Per-file titles/descriptions for museum archives

- Museum Mode
  - Bulk uploads (up to 5 files in a batch)
  - Persisted analysis results
  - Retrieval, detail view, and deletion endpoints

- Developer Friendly
  - Clear REST endpoints
  - .env configuration
  - MongoDB storage

---

## System Architecture

Services:
- API: Flask app (REST)
- DB: MongoDB
- AI Providers: OpenAI (e.g., GPT-4), Google Gemini
- Vectorization (optional): FAISS for semantic search
- File IO: Temp uploads with format-specific loaders

High-level flow:
1. Client uploads documents (auth required).
2. Backend validates, extracts text (PyPDF/docx2txt/plain).
3. AI pipelines run: emotions, locations, topics, writer info, people.
4. Results returned (individual) or saved (museum).
5. Museum endpoints manage archives.

---

## Getting Started

Prerequisites:
- Python 3.8+
- MongoDB (local or remote)
- API keys:
  - OPENAI_API_KEY
  - GOOGLE_API_KEY

Install:
- git clone <your-repository-url>
- cd HistoriSenseBackend
- python -m venv venv
- On Windows: venv\Scripts\activate
- On Linux/Mac: source venv/bin/activate
- pip install -r requirements.txt

---

## Configuration

Create a .env file in project root:

OPENAI_API_KEY=<your-openai-api-key>
GOOGLE_API_KEY=<your-google-ai-api-key>
MONGODB_URI=mongodb://localhost:27017/
JWT_SECRET_KEY=<your-jwt-secret>
MAX_CONTENT_LENGTH=16777216
MAX_MUSEUM_UPLOADS=5
ALLOWED_EXTENSIONS=pdf,docx,txt
DATABASE_NAME=HistoriSense

Notes:
- Use secure, unique values for JWT_SECRET_KEY.
- For cloud MongoDB, set MONGODB_URI accordingly (e.g., mongodb+srv://...).

---

## Running the Server

- Development:
  - python app.py
  - Default: http://localhost:5000

- Alternate run (if using Flask CLI):
  - set FLASK_APP=app.py (Windows PowerShell: $env:FLASK_APP="app.py")
  - flask run --port 5000

- Health check:
  - GET http://localhost:5000/ (or a dedicated /health if implemented)

---

## API Reference

Authentication

POST /api/signup
- Registers a user.
- Body:
  {
    "email": "user@example.com",
    "password": "SecurePass123!",
    "fullName": "John Doe",
    "userType": "individual"
  }
- Response: { "message": "User registered successfully" }

POST /api/login
- Authenticates and returns JWT.
- Body:
  {
    "email": "user@example.com",
    "password": "SecurePass123!"
  }
- Response:
  {
    "message": "Login successful",
    "access_token": "<jwt>",
    "success": true,
    "role": "individual",
    "name": "John Doe"
  }

Analysis

POST /api/analyze-testimony
- Upload and analyze one or more files.
- Headers: Authorization: Bearer <jwt>, Content-Type: multipart/form-data
- Form fields:
  - files: one or more files (pdf/docx/txt)
  - Optional per-file meta:
    - title_<filename>
    - description_<filename>
- Response (per file):
  {
    "message": "Testimony analyzed successfully",
    "analysis": {
      "filename": "war_memoir.pdf",
      "title": "War Memoir",
      "writer_info": { ... },
      "emotions": { "Fear": 35.2, ... },
      "locations": { "London": 15, "France": 12 },
      "topics": { "Combat Experience": 40.5, ... },
      "people_mentioned": [ { "name": "Captain Williams", "role": "CO", "region": "London" } ]
    }
  }

Museum Management

GET /api/museum-testimonies
- List stored testimonies (museum role).
- Headers: Authorization: Bearer <jwt>
- Response:
  {
    "testimonies": [
      {
        "filename": "testimony1.pdf",
        "title": "WWII Soldier Account",
        "upload_date": "2024-01-15T10:30:00",
        "file_type": ".pdf"
      }
    ]
  }

GET /api/museum-testimony/<filename>
- Retrieve a single testimony’s detailed analysis.

DELETE /api/museum-testimony/<filename>
- Delete a testimony.

Notes:
- Actual field names may vary slightly based on implementation. Check app.py for exact schemas if needed.

---

## Data & AI Components

Emotional Analysis
- Top emotions with percentage distribution.
- Example labels: Fear, Determination, Relief, Disgust, Hope, Anger, Sadness, Surprise, Neutral, Anxiety, Desperation, Gratitude.

Geographical Extraction
- Cities, regions, countries with mention counts and ranking.

Topic Classification
- Thematic distribution (e.g., Combat Experience, War Impact, Daily Life, Military Operations, Emigration Process, Civilian Interaction).

Writer Information Extraction
- Name, Country, Role, Age at time, Birth year, Death year (inferred if possible).

People Mentioned
- Other individuals (excluding the primary writer), roles, regions, and relationships where detectable.

---

## Database Schema

Users (collection: users)
- _id: ObjectId
- email: String (unique)
- password: String (bcrypt hash)
- fullName: String
- userType: String ("individual" | "museum")

Museum Testimonies (collection: museum_testimonies)
- _id: ObjectId
- filename: String
- title: String
- description: String
- writer_info: { Name, Country, Role, Age at time, Birth year, Death year }
- people_mentioned: [ { name, role, region } ]
- emotions: { emotion: Number }
- locations: { location: Number }
- topics: { topic: Number }
- upload_date: ISO String
- user_email: String
- file_type: String

---

## File Processing & Limits

Supported formats:
- PDF: PyPDF-based loader
- DOCX: docx2txt
- TXT: direct read

Pipeline:
1) Validate extension and size.
2) Sanitize filename and store temporarily.
3) Extract text by format loader.
4) Invoke AI workflows.
5) Persist results (museum) or return response (individual).
6) Clean up temporary files.

Limits and safety:
- MAX_CONTENT_LENGTH = 16MB
- MAX_MUSEUM_UPLOADS = 5 files per request
- Allowed extensions: pdf, docx, txt
- Path traversal protection and secure filename handling.

---

## Security & Best Practices

- JWT with expiration (e.g., 1 hour)
- bcrypt salted password hashing
- Validate email and password strength
- CORS configured appropriately
- Principle of least privilege for DB users in production
- Never commit secrets; use environment variables
- Sanitize file names and restrict types
- Log minimal PII and avoid storing raw documents when not required

---

## Troubleshooting

- 401 Unauthorized
  - Ensure Authorization: Bearer <jwt> is present and token not expired.

- 400 Bad Request on upload
  - Check file types and size. Ensure multipart/form-data and field name files.

- MongoDB connection issues
  - Verify MONGODB_URI, DB availability, and credentials.

- AI provider errors or slow responses
  - Verify OPENAI_API_KEY and GOOGLE_API_KEY.
  - Consider retry with backoff; check quota/limits.

- Unicode or parsing issues with PDFs
  - Try alternate PDF extraction tools or upload a cleaner copy if possible.

---

## Roadmap

- Add pagination and filters to museum testimonies.
- Webhooks or async processing for large batches.
- Export analysis as CSV/JSON for downstream research.
- Add confidence scores per AI section.
- Optional vector search endpoint for semantic queries.
- Rate limiting and audit logging.

---

## Contributing

This is an individual project by Vihanga Palihakkara. Suggestions and issue reports are welcome.

Guidelines:
- Follow PEP8 and conventional commit messages where possible.
- Include docstrings and error handling.
- Update README/API docs for endpoint changes.
- Add tests for new features.

---

## License

All rights reserved. This project is developed as an individual academic/research project by Vihanga Palihakkara. For usage inquiries, please contact the author.

---

## Acknowledgements

- OpenAI and Google Generative AI for LLM capabilities
- Flask and the Python OSS community
- MongoDB ecosystem

---

## Contact

Vihanga Palihakkara  
Associate Software Engineer  
Email: vihangawork@gmail.com

If you use HistoriSense in your research or demos, a quick note by email would make my day.
