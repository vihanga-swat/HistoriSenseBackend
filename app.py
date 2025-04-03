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
    return full_text

# Emotional Analysis
def analyze_emotions(full_text):
    analysis_text = full_text

    emotion_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
        Analyze the following war testimony and identify the top 5 emotions present.
        Focus on the emotional state of the writer and the emotional tone of the narrative.

        Consider emotions such as: Fear, Determination, Relief, Disgust, Hope, Anger, Sadness,
        Surprise, Neutral, Anxiety, Desperation, Gratitude, etc.

        Provide a percentage for each emotion that reflects its prominence in the text.
        The percentages should add up to 100% across the top 5 emotions.

        Text: {text}

        Return ONLY the top 5 emotions with their percentages in this exact format:
        - [Emotion]: [XX.X]%
        - [Emotion]: [XX.X]%
        - [Emotion]: [XX.X]%
        - [Emotion]: [XX.X]%
        - [Emotion]: [XX.X]%
        """
    )

    emotion_chain = LLMChain(llm=openai_llm, prompt=emotion_prompt_template)
    result = emotion_chain.run(text=analysis_text)

    # Parse the emotions
    emotions = {}
    for line in result.strip().split('\n'):
        if line.strip():
            try:
                parts = line.split(': ')
                if len(parts) == 2:
                    emotion = parts[0].strip('- ')
                    percentage = float(parts[1].strip('%'))
                    emotions[emotion] = percentage
            except (ValueError, IndexError):
                continue

    return emotions

# Geographical Analysis
def extract_locations(full_text):
    analysis_text = full_text

    location_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
        Extract all geographical locations (cities, countries, regions, specific places) mentioned in this war testimony.
        Count the number of times each location is mentioned.
        Focus only on real geographical places, not generic terms.

        Text: {text}

        Return the locations in descending order of mentions in this exact format:
        1. [Location]: [X] mentions
        2. [Location]: [X] mentions
        3. [Location]: [X] mentions
        ...and so on
        """
    )

    location_chain = LLMChain(llm=google_llm, prompt=location_prompt_template)
    result = location_chain.run(text=analysis_text)

    # Parse the locations
    locations = {}
    for line in result.strip().split('\n'):
        if line.strip():
            try:
                parts = line.split(': ')
                if len(parts) == 2:
                    location_with_num = parts[0].strip()
                    location = location_with_num.split('. ')[1] if '. ' in location_with_num else location_with_num
                    count_part = parts[1].split(' ')[0]
                    count = int(count_part)
                    locations[location] = count
            except (ValueError, IndexError):
                continue

    return locations

# Key Topics Extraction
def extract_key_topics(full_text):
    analysis_text = full_text

    topic_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
        Analyze this war testimony and identify the top 5 key topics or themes.
        Consider topics such as: Combat Experience, Emigration Process, War Impact, Daily Life,
        Military Operations, Civilian Interaction, etc.

        For each topic, calculate a percentage that represents how prominent it is in the testimony.
        The percentages should add up to 100% across the top 5 topics.

        Text: {text}

        Return ONLY the top 5 topics with their percentages in this exact format:
        1. [Topic]: [XX.X]%
        2. [Topic]: [XX.X]%
        3. [Topic]: [XX.X]%
        4. [Topic]: [XX.X]%
        5. [Topic]: [XX.X]%
        """
    )

    topic_chain = LLMChain(llm=openai_llm, prompt=topic_prompt_template)
    result = topic_chain.run(text=analysis_text)

    # Parse the topics
    topics = {}
    for line in result.strip().split('\n'):
        if line.strip():
            try:
                parts = line.split(': ')
                if len(parts) == 2:
                    topic_with_num = parts[0].strip()
                    topic = topic_with_num.split('. ')[1] if '. ' in topic_with_num else topic_with_num
                    percentage = float(parts[1].strip('%'))
                    topics[topic] = percentage
            except (ValueError, IndexError):
                continue

    return topics

# Updated extract_testimony_details
def extract_testimony_details(file_path):
    full_text = process_document(file_path)

    # Use a maximum of 15000 characters to avoid token limits
    analysis_text = full_text

    # Writer details
    writer_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
        Extract the following details about the writer of this war testimony:
        - Full Name
        - Country of origin
        - Role during the war (e.g., Civilian, Military, etc.)
        - Age at the time of events described
        - Birth year (calculate if possible)
        - Death year (if mentioned)

        If any detail is not explicitly mentioned, make a reasonable inference based on the text.
        If you cannot determine a detail even through inference, state "Not specified".

        Text: {text}

        Return ONLY these details in this exact format:
        Name: [value]
        Country: [value]
        Role: [value]
        Age at time: [value]
        Birth year: [value]
        Death year: [value]
        """
    )

    writer_chain = LLMChain(llm=google_llm, prompt=writer_prompt_template)
    writer_result = writer_chain.run(text=analysis_text)

    # People mentioned
    people_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
        Extract all people mentioned in this war testimony (excluding the writer).
        For each person, identify:
        - Their full name
        - Their role or relationship to the writer
        - The region/location they are associated with

        Text: {text}

        Return ONLY these details in this exact format:
        - Name: [name], Role: [role], Region: [region]
        - Name: [name], Role: [role], Region: [region]
        - Name: [name], Role: [role], Region: [region]
        - Name: [name], Role: [role], Region: [region]
        ...and so on
        """
    )

    people_chain = LLMChain(llm=openai_llm, prompt=people_prompt_template)
    people_result = people_chain.run(text=analysis_text)

    # Emotional, geographical, and topic analysis
    emotions = analyze_emotions(full_text)
    locations = extract_locations(full_text)
    topics = extract_key_topics(full_text)

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

if __name__ == '__main__':
    app.run(debug=True)