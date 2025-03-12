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
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Google Gemini API setup
GOOGLE_API_KEY = "AIzaSyBzYZ17YOlHDBYnOLFatNinb7GujbAgCYc"  # Replace with your actual API key
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.2)

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

# New route for testimony analysis without database storage
@app.route('/api/analyze-testimony', methods=['POST'])
@jwt_required()
def analyze_testimony():
    current_user_email = get_jwt_identity()

    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Process the uploaded PDF
            writer_info, people_info = extract_testimony_details(file_path)

            # Parse writer info into structured data
            writer_data = parse_writer_info(writer_info)
            people_data = parse_people_info(people_info)

            # Log the parsed data
            app.logger.info(f"Parsed writer data: {writer_data}")
            app.logger.info(f"Parsed people data: {people_data}")

            # Store analysis results in session instead of database
            analysis_result = {
                "filename": filename,
                "writer_info": writer_data,
                "people_mentioned": people_data,
                "analysis_date": datetime.now().isoformat()
            }

            # Clean up the file after processing
            try:
                os.remove(file_path)
            except Exception as e:
                app.logger.warning(f"Could not remove temporary file: {str(e)}")

            return jsonify({
                "message": "Testimony analyzed successfully",
                "analysis": analysis_result
            }), 200

        except Exception as e:
            # Clean up on error
            try:
                os.remove(file_path)
            except:
                pass
            return jsonify({"error": f"Error processing testimony: {str(e)}"}), 500

    return jsonify({"error": "File type not allowed"}), 400

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

def extract_testimony_details(pdf_path):
    # Load and extract text from the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    full_text = "\n".join([doc.page_content for doc in documents])

    # Define prompt for extracting writer details
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
        - If the owner is a soldier, their unit

        Text: {text}

        Return the result in this format:
        Name: [value]
        Country: [value]
        Role: [value]
        Age at time: [value]
        Birth year: [value]
        Death year: [value]
        Unit (if soldier): [value]
        """
    )

    # Create and run the LLM chain for writer details
    writer_chain = LLMChain(llm=llm, prompt=writer_prompt_template)
    writer_result = writer_chain.run(text=full_text)

    # Define prompt for extracting mentioned people
    people_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
        You are an expert in analyzing war testimonies. Identify all people mentioned in the text, along with their roles (e.g., soldier, commander, civilian) and regions (e.g., country, city, or battlefield location) where they are associated. Ensure high accuracy by relying on explicit mentions and strong contextual evidence. If a role or region is unclear, mark it as "Unspecified". Ignore generic references (e.g., "the soldiers") and focus on named individuals or specific roles (like "the commander"). If a name appears multiple times, only list it once with the most relevant role and region. don't mention peoples in the references or notes section and don't mention again the owner's name of the testimony.

        Text: {text}

        Return the result in EXACTLY this format for EACH person (one person per line):
        - Name: [person name], Role: [person role], Region: [person region]
        must Do not deviate from this format. Do not add extra text or explanations.
        """
    )

    # Create and run the LLM chain for people mentioned
    people_chain = LLMChain(llm=llm, prompt=people_prompt_template)
    people_result = people_chain.run(text=full_text)

    print("people_result:", people_result)

    return writer_result, people_result

def parse_writer_info(writer_info):
    # Parse the writer info string into a structured dictionary
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
            # Remove the leading dash
            line = line.strip()[1:].strip()

            # Initialize person data
            person = {
                "name": "Unspecified",
                "role": "Unspecified",
                "region": "Unspecified"
            }

            # Extract name (everything before "Role:")
            if "Role:" in line:
                name = line.split("Role:")[0].strip()
                # Remove any trailing comma
                if name.endswith(','):
                    name = name[:-1].strip()
                person["name"] = name
            else:
                # If no "Role:" marker, just use the whole line as name
                person["name"] = line

            # Extract role
            if "Role:" in line:
                role_part = line.split("Role:")[1]
                if "Region:" in role_part:
                    role = role_part.split("Region:")[0].strip()
                    # Remove any trailing comma
                    if role.endswith(','):
                        role = role[:-1].strip()
                    person["role"] = role
                else:
                    person["role"] = role_part.strip()

            # Extract region
            if "Region:" in line:
                region = line.split("Region:")[1].strip()
                person["region"] = region

            # Add to the list if we have a name
            if person["name"] and person["name"] != "Unspecified":
                people_data.append(person)

    # Debug logging
    print(f"Extracted {len(people_data)} people from the testimony")
    for person in people_data:
        print(f"Person: {person}")

    return people_data

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