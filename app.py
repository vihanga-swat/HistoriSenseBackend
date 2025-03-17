from flask import Flask, request, jsonify, session
from flask_cors import CORS
from pymongo import MongoClient
import bcrypt
import re
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from datetime import timedelta, datetime
import secrets
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from collections import Counter

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Enable CORS with credentials support
app.secret_key = secrets.token_urlsafe(32)  # For session management

# Setup the Flask-JWT-Extended extension
app.config['JWT_SECRET_KEY'] = secrets.token_urlsafe(64)  # Generate a secure secret key
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)  # Tokens expire after 1 hour
jwt = JWTManager(app)

# MongoDB Connection
client = MongoClient('localhost', 27017)
db = client['HistoriSense']
users = db.users
museum_testimonies = db.museum_testimonies

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
MAX_MUSEUM_UPLOADS = 5

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Dual LLMs and Embeddings
openai_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.5,
    openai_api_key=OPENAI_API_KEY
)
google_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.5
)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Helper functions
def validate_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def validate_password(password):
    return bool(re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=\[\]{};:\'",.<>?]).{8,}$', password))

def validate_full_name(full_name):
    return len(full_name.split()) >= 2

def validate_user_type(user_type):
    return user_type in ['individual', 'museum']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Existing routes
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    full_name = data.get('fullName')
    user_type = data.get('userType')

    # Server-side validation
    if not validate_email(email):
        return jsonify({"error": "Invalid email format"}), 400
    if not validate_password(password):
        return jsonify({"error": "Password does not meet complexity requirements"}), 400
    if not validate_full_name(full_name):
        return jsonify({"error": "Please provide a valid full name"}), 400
    if not validate_user_type(user_type):
        return jsonify({"error": "Invalid user type"}), 400

    # Check if user already exists
    if users.find_one({'email': email}):
        return jsonify({"error": "Email already registered"}), 400

    # Hash the password
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Insert user into database
    user = {
        'email': email,
        'password': hashed,
        'fullName': full_name,
        'userType': user_type
    }
    users.insert_one(user)

    return jsonify({"message": "User registered successfully"}), 201

@app.route('/api/login', methods=['POST'])
def login():
    email = request.json.get('email')
    password = request.json.get('password')

    user = users.find_one({'email': email})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        # Create access token with user identity
        access_token = create_access_token(identity=email)
        # Return user details including role for frontend routing
        return jsonify({
            "message": "Login successful",
            "access_token": access_token,
            "success": True,
            "role": user['userType'],  # 'museum' or 'individual'
            "name": user['fullName']   # Include user's full name
        }), 200
    return jsonify({"error": "Invalid credentials", "success": False}), 401

# Optional: Protected route example to verify JWT and user identity
# @app.route('/api/protected', methods=['GET'])
# @jwt_required()
# def protected():
#     current_user_email = get_jwt_identity()
#     user = users.find_one({'email': current_user_email})
#     if user:
#         return jsonify({
#             "message": "Access granted",
#             "email": current_user_email,
#             "role": user['userType'],
#             "name": user['fullName']
#         }), 200
#     return jsonify({"error": "User not found"}), 404

# Document processing and chunk retrieval
def process_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    documents = loader.load()
    full_text = "\n".join([doc.page_content for doc in documents])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    vector_db = FAISS.from_documents(chunks, embeddings)
    return full_text, vector_db

def retrieve_relevant_chunks(vector_db, query, k=3):
    """Retrieves the top k relevant document chunks for a given query."""
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in relevant_docs])

# Emotional Analysis
def analyze_emotions_with_llm(vector_db):
    emotion_query = "Text containing strong emotional content (e.g., anger, fear, sadness)"
    relevant_text = retrieve_relevant_chunks(vector_db, emotion_query, k=3)
    chunks = [relevant_text[i:i+500] for i in range(0, len(relevant_text), 500)]
    
    emotion_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
        Analyze the following text and identify the top 5 emotions present (e.g., anger, fear, joy, sadness, surprise, disgust, neutral). 
        Provide a percentage score for each emotion based on your interpretation. Return only the top 5 emotions with their scores.

        Text: {text}

        Return the result in this format:
        - [emotion]: [score]%
        """
    )
    
    emotion_chain = LLMChain(llm=openai_llm, prompt=emotion_prompt_template)
    all_emotions = []
    for chunk in chunks:
        if chunk.strip():
            result = emotion_chain.run(text=chunk)
            all_emotions.append(result)
    
    emotion_scores = {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'neutral': 0, 'sadness': 0, 'surprise': 0}
    count = 0
    for result in all_emotions:
        lines = result.strip().split('\n')
        for line in lines:
            if line.strip():
                try:
                    emotion, score = line.split(': ')
                    emotion = emotion.strip('- ').lower()
                    score = float(score.strip('%'))
                    if emotion in emotion_scores:
                        emotion_scores[emotion] += score
                        count += 1
                except ValueError:
                    continue
    if count > 0:
        for emotion in emotion_scores:
            emotion_scores[emotion] /= count
    top_emotions = dict(sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:5])
    return top_emotions

# Geographical Analysis
def extract_locations_with_llm(vector_db):
    location_query = "Text containing geographical locations (e.g., cities, countries, regions)"
    relevant_text = retrieve_relevant_chunks(vector_db, location_query, k=3)
    chunks = [relevant_text[i:i+500] for i in range(0, len(relevant_text), 500)]
    
    location_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
        Extract all geographical locations (e.g., cities, countries, regions) mentioned in the text along with associated events or descriptions if present. 
        For each location, provide a count of mentions and a brief description of the context (e.g., "Battle occurred here"). If no specific event is mentioned, use "Mentioned in context".

        Text: {text}

        Return the result in this format:
        - [location]: [count], [description]
        """
    )
    
    location_chain = LLMChain(llm=google_llm, prompt=location_prompt_template)
    all_locations = []
    for chunk in chunks:
        if chunk.strip():
            result = location_chain.run(text=chunk)
            all_locations.append(result)
    
    location_data = {}
    for result in all_locations:
        lines = result.strip().split('\n')
        for line in lines:
            if line.strip():
                try:
                    parts = line.split(': ', 1)
                    if len(parts) < 2:
                        continue
                    location_info = parts[1].split(', ', 1)
                    location = parts[0].strip('- ')
                    count = int(location_info[0].strip())
                    description = location_info[1].strip() if len(location_info) > 1 else "Mentioned in context"
                    if location in location_data:
                        location_data[location]['count'] += count
                    else:
                        location_data[location] = {'count': count, 'description': description}
                except (ValueError, IndexError):
                    continue
    return location_data

# Key Topics Extraction
def extract_key_topics_with_llm(vector_db):
    topic_query = "Text containing thematic content (e.g., military operations, civilian interactions)"
    relevant_text = retrieve_relevant_chunks(vector_db, topic_query, k=3)
    chunks = [relevant_text[i:i+500] for i in range(0, len(relevant_text), 500)]
    
    topic_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
        Analyze the following text and identify mentions of the following key topics: Military Operations, Civilian Interaction, War Impact, Daily Life, and Combat Experience. 
        Count how many times each topic is referenced (either explicitly or contextually). Return the counts for each topic.

        Text: {text}

        Return the result in this format:
        - [topic]: [count]
        """
    )
    
    topic_chain = LLMChain(llm=openai_llm, prompt=topic_prompt_template)
    all_topics = []
    for chunk in chunks:
        if chunk.strip():
            result = topic_chain.run(text=chunk)
            all_topics.append(result)
    
    topic_counter = Counter({
        "Military Operations": 0,
        "Civilian Interaction": 0,
        "War Impact": 0,
        "Daily Life": 0,
        "Combat Experience": 0
    })
    for result in all_topics:
        lines = result.strip().split('\n')
        for line in lines:
            if line.strip():
                try:
                    topic, count = line.split(': ')
                    topic = topic.strip('- ')
                    count = int(count.strip())
                    if topic in topic_counter:
                        topic_counter[topic] += count
                except ValueError:
                    continue
    return dict(topic_counter)

# Updated extract_testimony_details
def extract_testimony_details(file_path):
    full_text, vector_db = process_document(file_path)
    
    # Writer details
    writer_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
        You are an expert in analyzing war testimonies. Extract the following details about the owner from the text with high accuracy. If a detail is not explicitly mentioned, return "Not specified". Use context clues where possible but prioritize explicit mentions for accuracy:

        - Name
        - Country
        - Role (e.g., soldier, civilian, nurse)
        - Age at time of testimony
        - Birth year
        - Death year

        Text: {text}

        Return the result in this format:
        Name: [value]
        Country: [value]
        Role: [value]
        Age at time: [value]
        Birth year: [value]
        Death year: [value]
        """
    )
    writer_query = "Details about the writer or owner of this testimony"
    writer_relevant_text = retrieve_relevant_chunks(vector_db, writer_query, k=3)
    writer_chain = LLMChain(llm=google_llm, prompt=writer_prompt_template)
    writer_result = writer_chain.run(text=writer_relevant_text)

    # People mentioned
    people_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
        You are an expert in analyzing war testimonies. Identify all people mentioned in the text, along with their roles (e.g., soldier, commander, civilian) and regions (e.g., country, city, or battlefield location) where they are associated. Ensure high accuracy by relying on explicit mentions and strong contextual evidence. If a role or region is unclear, mark it as "Unspecified". Ignore generic references (e.g., "the soldiers") and focus on named individuals. If a name appears multiple times, only list it once with the most relevant role and region.

        Text: {text}

        Return the result in this format:
        - Name: [value], Role: [value], Region: [value]
        """
    )
    people_query = "People mentioned in the testimony with their roles and regions"
    people_relevant_text = retrieve_relevant_chunks(vector_db, people_query, k=3)
    people_chain = LLMChain(llm=openai_llm, prompt=people_prompt_template)
    people_result = people_chain.run(text=people_relevant_text)

    # Emotional, geographical, and topic analysis
    emotions = analyze_emotions_with_llm(vector_db)
    locations = extract_locations_with_llm(vector_db)
    topics = extract_key_topics_with_llm(vector_db)

    return writer_result, people_result, emotions, locations, topics

# Parsing functions (unchanged)
def parse_writer_info(writer_info):
    lines = writer_info.strip().split('\n')
    writer_data = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            writer_data[key.strip()] = value.strip()
    return writer_data

def parse_people_info(people_info):
    people_data = []
    lines = people_info.strip().split('\n')
    for line in lines:
        if line.strip().startswith('-'):
            name_match = re.search(r'Name:\s*\[(.*?)\]|Name:\s*(.*?)(?:,|$)', line)
            role_match = re.search(r'Role:\s*\[(.*?)\]|Role:\s*(.*?)(?:,|$)', line)
            region_match = re.search(r'Region:\s*\[(.*?)\]|Region:\s*(.*?)(?:,|$)', line)
            name = next((g for g in name_match.groups() if g is not None), None) if name_match else None
            if name:
                role = next((g for g in role_match.groups() if g is not None), "Unspecified") if role_match else "Unspecified"
                region = next((g for g in region_match.groups() if g is not None), "Unspecified") if region_match else "Unspecified"
                people_data.append({"name": name.strip(), "role": role.strip(), "region": region.strip()})
    return people_data

# Updated analyze-testimony route
@app.route('/api/analyze-testimony', methods=['POST'])
@jwt_required()
def analyze_testimony():
    current_user_email = get_jwt_identity()
    user = users.find_one({'email': current_user_email})
    user_type = user['userType']

    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected files"}), 400

    if user_type == 'museum' and len(files) > MAX_MUSEUM_UPLOADS:
        return jsonify({"error": f"Maximum {MAX_MUSEUM_UPLOADS} files allowed for museum users"}), 400
    elif user_type == 'individual' and len(files) > 1:
        return jsonify({"error": "Individual users can only upload one file at a time"}), 400

    analysis_results = []
    files_to_remove = []

    try:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                files_to_remove.append(file_path)

                # Process and analyze the file
                writer_info, people_info, emotions, locations, topics = extract_testimony_details(file_path)
                writer_data = parse_writer_info(writer_info)
                people_data = parse_people_info(people_info)

                analysis_result = {
                    "filename": filename,
                    "title": request.form.get(f'title_{filename}', filename),
                    "description": request.form.get(f'description_{filename}', ''),
                    "writer_info": writer_data,
                    "people_mentioned": people_data,
                    "emotions": emotions,
                    "locations": locations,
                    "topics": topics,
                    "upload_date": datetime.now().isoformat(),
                    "user_email": current_user_email,
                    "file_type": '.' + filename.rsplit('.', 1)[1].lower()
                }

                if user_type == 'museum':
                    museum_testimonies.insert_one(analysis_result)
                else:
                    analysis_results.append(analysis_result)

        for file_path in files_to_remove:
            try:
                os.remove(file_path)
            except Exception as e:
                app.logger.warning(f"Could not remove temporary file: {str(e)}")

        if user_type == 'museum':
            return jsonify({"message": "Testimonies analyzed and stored successfully"}), 200
        else:
            return jsonify({
                "message": "Testimony analyzed successfully",
                "analysis": analysis_results[0]
            }), 200

    except Exception as e:
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
            except:
                pass
        return jsonify({"error": f"Error processing testimonies: {str(e)}"}), 500

# New routes for testimony analysis
# @app.route('/api/upload-testimony', methods=['POST'])
# @jwt_required()
# def upload_testimony():
#     current_user_email = get_jwt_identity()

#     # Check if the post request has the file part
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['file']

#     # If user does not select file, browser also submits an empty part without filename
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         try:
#             # Process the uploaded PDF
#             writer_info, people_info = extract_testimony_details(file_path)

#             # Log the raw responses for debugging
#             app.logger.info(f"Raw writer info: {writer_info}")
#             app.logger.info(f"Raw people info: {people_info}")

#             # Parse writer info into structured data
#             writer_data = parse_writer_info(writer_info)
#             people_data = parse_people_info(people_info)

#             # Log the parsed data
#             app.logger.info(f"Parsed writer data: {writer_data}")
#             app.logger.info(f"Parsed people data: {people_data}")

#             # Save to database
#             testimony_data = {
#                 "user_email": current_user_email,
#                 "filename": filename,
#                 "file_path": file_path,
#                 "writer_info": writer_data,
#                 "people_mentioned": people_data
#             }

#             result = testimonies.insert_one(testimony_data)

#             return jsonify({
#                 "message": "Testimony uploaded and analyzed successfully",
#                 "testimony_id": str(result.inserted_id),
#                 "writer_info": writer_data,
#                 "people_mentioned": people_data
#             }), 201

#         except Exception as e:
#             return jsonify({"error": f"Error processing testimony: {str(e)}"}), 500

#     return jsonify({"error": "File type not allowed"}), 400

# New route to get museum testimonies list
@app.route('/api/museum-testimonies', methods=['GET'])
@jwt_required()
def get_museum_testimonies():
    current_user_email = get_jwt_identity()
    user = users.find_one({'email': current_user_email})
    
    if user['userType'] != 'museum':
        return jsonify({"error": "Unauthorized access"}), 403

    testimonies = list(museum_testimonies.find(
        {'user_email': current_user_email},
        {'_id': 0, 'filename': 1, 'title': 1, 'upload_date': 1, 'file_type': 1}
    ))
    
    # Add file_type based on filename extension
    for testimony in testimonies:
        testimony['file_type'] = '.' + testimony['filename'].rsplit('.', 1)[1].lower()
    
    return jsonify({"testimonies": testimonies}), 200

# New route to get specific testimony details
@app.route('/api/museum-testimony/<filename>', methods=['GET'])
@jwt_required()
def get_museum_testimony(filename):
    current_user_email = get_jwt_identity()
    user = users.find_one({'email': current_user_email})
    
    if user['userType'] != 'museum':
        return jsonify({"error": "Unauthorized access"}), 403

    testimony = museum_testimonies.find_one(
        {'user_email': current_user_email, 'filename': filename},
        {'_id': 0}
    )
    
    if not testimony:
        return jsonify({"error": "Testimony not found"}), 404
        
    return jsonify({"testimony": testimony}), 200

# New route to delete a specific testimony
@app.route('/api/museum-testimony/<filename>', methods=['DELETE'])
@jwt_required()
def delete_museum_testimony(filename):
    current_user_email = get_jwt_identity()
    user = users.find_one({'email': current_user_email})
    
    if user['userType'] != 'museum':
        return jsonify({"error": "Unauthorized access"}), 403

    # Check if the testimony exists and belongs to the user
    result = museum_testimonies.delete_one({
        'user_email': current_user_email,
        'filename': filename
    })

    if result.deleted_count == 0:
        return jsonify({"error": "Testimony not found or you don't have permission to delete it"}), 404

    return jsonify({"message": "Testimony deleted successfully"}), 200

# def extract_testimony_details(file_path):
#     # Process the document
#     full_text, vector_db = process_document(file_path)

#     # Define prompt for extracting writer details
#     writer_prompt_template = PromptTemplate(
#         input_variables=["text"],
#         template="""
#         You are an expert in analyzing war testimonies. Extract the following details about the owner from the text with high accuracy. If a detail is not explicitly mentioned, return "Not specified". Use context clues where possible but prioritize explicit mentions for accuracy:

#         - Name
#         - Country
#         - Role (e.g., soldier, civilian, nurse)
#         - Age at time of testimony
#         - Birth year
#         - Death year

#         Text: {text}

#         Return the result in this format:
#         Name: [value]
#         Country: [value]
#         Role: [value]
#         Age at time: [value]
#         Birth year: [value]
#         Death year: [value]
#         """
#     )

#     # Writer extraction
#     writer_query = "Details about the writer or owner of this testimony"
#     writer_relevant_text = retrieve_relevant_chunks(vector_db, writer_query, k=3)
#     writer_chain = LLMChain(llm=google_llm, prompt=writer_prompt_template)
#     writer_result = writer_chain.run(text=writer_relevant_text)

#     # Define prompt for extracting mentioned people
#     people_prompt_template = PromptTemplate(
#         input_variables=["text"],
#         template="""
#         You are an expert in analyzing war testimonies. Identify all people mentioned in the text, along with their roles (e.g., soldier, commander, civilian) and regions (e.g., country, city, or battlefield location) where they are associated. Ensure high accuracy by relying on explicit mentions and strong contextual evidence. If a role or region is unclear, mark it as "Unspecified". Ignore generic references (e.g., "the soldiers") and focus on named individuals. If a name appears multiple times, only list it once with the most relevant role and region.

#         Text: {text}

#         Return the result in this format:
#         - Name: [value], Role: [value], Region: [value]
#         """
#     )

#     # People extraction
#     people_query = "People mentioned in the testimony with their roles and regions"
#     people_relevant_text = retrieve_relevant_chunks(vector_db, people_query, k=3)
#     people_chain = LLMChain(llm=openai_llm, prompt=people_prompt_template)
#     people_result = people_chain.run(text=people_relevant_text)

#     return writer_result, people_result

# def parse_writer_info(writer_info):
#     # Parse the writer info string into a structured dictionary
#     lines = writer_info.strip().split('\n')
#     writer_data = {}

#     for line in lines:
#         if ':' in line:
#             key, value = line.split(':', 1)
#             writer_data[key.strip()] = value.strip()

#     return writer_data

# def parse_people_info(people_info):
#     people_data = []
#     lines = people_info.strip().split('\n')

#     for line in lines:
#         # Look for lines that contain information about a person
#         if line.strip().startswith('-'):
#             # Try different regex patterns to handle variations in format
#             name_match = re.search(r'Name:\s*\[(.*?)\]|Name:\s*(.*?)(?:,|$)', line)
#             role_match = re.search(r'Role:\s*\[(.*?)\]|Role:\s*(.*?)(?:,|$)', line)
#             region_match = re.search(r'Region:\s*\[(.*?)\]|Region:\s*(.*?)(?:,|$)', line)

#             name = None
#             if name_match:
#                 # Get the first non-None group
#                 name = next((g for g in name_match.groups() if g is not None), None)

#             if name:  # Only add if we found a name
#                 role = next((g for g in role_match.groups() if g is not None), "Unspecified") if role_match else "Unspecified"
#                 region = next((g for g in region_match.groups() if g is not None), "Unspecified") if region_match else "Unspecified"

#                 person = {
#                     "name": name.strip(),
#                     "role": role.strip(),
#                     "region": region.strip()
#                 }
#                 people_data.append(person)

#     # If no people were found with the above method, try a simpler approach
#     if not people_data and people_info.strip():
#         # Just extract any lines that look like they contain person information
#         for line in lines:
#             if line.strip() and not line.strip().startswith('#') and ':' in line:
#                 person = {"description": line.strip()}
#                 people_data.append(person)

#     return people_data

# @app.route('/api/testimonies', methods=['GET'])
# @jwt_required()
# def get_testimonies():
#     current_user_email = get_jwt_identity()
#     user = users.find_one({'email': current_user_email})

#     # For museum users, return all testimonies
#     # For individual users, return only their own testimonies
#     if user['userType'] == 'museum':
#         testimony_list = list(testimonies.find({}))
#     else:
#         testimony_list = list(testimonies.find({"user_email": current_user_email}))

#     # Convert ObjectId to string for JSON serialization
#     for testimony in testimony_list:
#         testimony['_id'] = str(testimony['_id'])

#     return jsonify(testimony_list), 200

# @app.route('/api/testimony/<testimony_id>', methods=['GET'])
# @jwt_required()
# def get_testimony(testimony_id):
#     from bson.objectid import ObjectId

#     testimony = testimonies.find_one({"_id": ObjectId(testimony_id)})

#     if not testimony:
#         return jsonify({"error": "Testimony not found"}), 404

#     # Convert ObjectId to string for JSON serialization
#     testimony['_id'] = str(testimony['_id'])

#     return jsonify(testimony), 200

if __name__ == '__main__':
    app.run(debug=True)