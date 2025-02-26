from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import bcrypt
import re
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from datetime import timedelta
import secrets

app = Flask(__name__)
CORS(app)  # Enable CORS for React app

# Setup the Flask-JWT-Extended extension
app.config['JWT_SECRET_KEY'] = secrets.token_urlsafe(64)  # Generate a secure secret key
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)  # Tokens expire after 1 hour
jwt = JWTManager(app)

# MongoDB Connection
client = MongoClient('localhost', 27017)
db = client['HistoriSense']
users = db.users

def validate_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def validate_password(password):
    return bool(re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=\[\]{};:\'",.<>?]).{8,}$', password))

def validate_full_name(full_name):
    return len(full_name.split()) >= 2

def validate_user_type(user_type):
    return user_type in ['individual', 'museum']

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

if __name__ == '__main__':
    app.run(debug=True)