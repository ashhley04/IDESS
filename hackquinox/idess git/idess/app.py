from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from docx import Document
import PyPDF2
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import json

app = Flask(__name__)
CORS(app)

# Set the folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set the folder to store JSON files
JSON_FOLDER = 'json_data'
app.config['JSON_FOLDER'] = JSON_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Helper Functions for File Handling and Text Extraction
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
        return text
    except Exception as e:
        return str(e)

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text
        return text
    except Exception as e:
        return str(e)

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
        return text
    except Exception as e:
        return str(e)

def extract_text_from_resume(file_path, extension):
    if extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif extension == 'docx':
        return extract_text_from_docx(file_path)
    elif extension == 'txt':
        return extract_text_from_txt(file_path)
    return ""


# Stage 1: Resume Analysis
def analyze_resume(resume_text):
    # Placeholder function to simulate resume analysis
    analysis_result = {
        'skills': ['Python', 'Data Analysis', 'Machine Learning', 'Deep Learning', 'Data Engineering'],
        'experience': 2,  # years of experience
        'swot': {
            'strengths': ['Problem-solving', 'Critical Thinking', 'Data Analysis'],
            'weaknesses': ['Time management', 'Public Speaking'],
            'opportunities': ['AI development', 'Data Science', 'Cloud Computing'],
            'threats': ['Automation', 'Outsourcing', 'Economic downturn']
        },
        'job_suggestions': 'Software Developer, Data Scientist, Data Engineer'
    }
    return analysis_result


# Stage 2: Course Suggestions based on Skills (Web scraping version)
def scrape_courses():
    url = 'https://www.coursera.org/browse/data-science'
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    courses = []
    
    # Example: Find all courses (you can modify the selectors to match the actual structure of the website you're scraping)
    course_elements = soup.find_all('a', {'class': 'card-title'})
    for element in course_elements[:10]:  # Get top 10 courses for more suggestions
        courses.append({
            'name': element.get_text(),
            'url': 'https://www.coursera.org' + element.get('href')
        })
    
    return courses

# Example function to fetch industry contacts (LinkedIn profiles)
def get_industry_contacts():
    contacts = [
        {'name': 'John Doe', 'role': 'Data Scientist', 'linkedin': 'https://www.linkedin.com/in/johndoe/'},
        {'name': 'Jane Smith', 'role': 'AI Researcher', 'linkedin': 'https://www.linkedin.com/in/janesmith/'},
        {'name': 'Mike Johnson', 'role': 'ML Expert', 'linkedin': 'https://www.linkedin.com/in/mikejohnson/'},
        {'name': 'Alice Brown', 'role': 'Data Engineer', 'linkedin': 'https://www.linkedin.com/in/alicebrown/'},
        {'name': 'Sam Green', 'role': 'Cloud Architect', 'linkedin': 'https://www.linkedin.com/in/samgreen/'}
    ]
    return contacts


# Load GPT-2 for text generation
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def get_suggestions_from_resume(resume_text):
    inputs = tokenizer.encode(resume_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Read and write JSON data for resume analysis
def save_to_json(filename, data):
    if not os.path.exists(app.config['JSON_FOLDER']):
        os.makedirs(app.config['JSON_FOLDER'])
    
    file_path = os.path.join(app.config['JSON_FOLDER'], filename)
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def load_from_json(filename):
    file_path = os.path.join(app.config['JSON_FOLDER'], filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    return {}


# Stage 1 - Resume Analysis Route
@app.route('/analyze_resume', methods=['POST'])
def analyze_resume_route():
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['resume']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Extract text from the file based on its extension
            file_extension = filename.rsplit('.', 1)[1].lower()
            resume_text = extract_text_from_resume(file_path, file_extension)

            if not resume_text:
                return jsonify({'error': 'Failed to extract text from the resume'}), 500

            # Analyze the resume
            result = analyze_resume(resume_text)

            # Save analysis result to JSON
            save_to_json(f"{filename}_analysis.json", result)

            # Return the result in JSON format
            return jsonify(result)

        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Stage 2 - Action Development Planning: AI Suggestions based on Resume Analysis
@app.route('/stage2/ai-suggestions', methods=['POST'])
def ai_suggestions():
    try:
        data = request.json

        # Ensure 'stage1_output' and 'section' are passed in the request
        stage1_output = data.get('stage1_output')
        section = data.get('section')

        if not stage1_output or not section:
            return jsonify({"error": "Stage 1 output and section are required"}), 400

        skills = stage1_output.get('skills', [])
        swot = stage1_output.get('swot', {})
        suggestions = []

        # Get course suggestions based on the skills from Stage 1
        for skill in skills:
            courses = scrape_courses()
            suggestions.append({
                "skill": skill,
                "courses": courses
            })

        # Add suggestions based on SWOT analysis if available
        swot_suggestions = []
        if 'opportunities' in swot:
            swot_suggestions.append({
                "section": "Opportunities",
                "suggestions": swot['opportunities']
            })
        if 'strengths' in swot:
            swot_suggestions.append({
                "section": "Strengths",
                "suggestions": swot['strengths']
            })
        if 'weaknesses' in swot:
            swot_suggestions.append({
                "section": "Weaknesses",
                "suggestions": swot['weaknesses']
            })
        if 'threats' in swot:
            swot_suggestions.append({
                "section": "Threats",
                "suggestions": swot['threats']
            })

        # Get industry contacts with LinkedIn profiles
        industry_contacts = get_industry_contacts()

        # Example career goals (you can customize this based on the input data)
        goals = ['Become a Data Scientist', 'Master Machine Learning', 'Improve Time Management', 'Learn Cloud Computing', 'Advance in AI Research']

        # Save suggestions to JSON
        save_to_json('stage2_suggestions.json', {
            "suggestions": suggestions,
            "swot_suggestions": swot_suggestions,
            "goals": goals,
            "industry_contacts": industry_contacts
        })

        # Return a structured response with success status
        return jsonify({
            "success": True,
            "suggestions": suggestions,
            "swot_suggestions": swot_suggestions,
            "goals": goals,
            "industry_contacts": industry_contacts
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    # Ensure that the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Ensure that the JSON folder exists
    if not os.path.exists(app.config['JSON_FOLDER']):
        os.makedirs(app.config['JSON_FOLDER'])

    # Run the Flask app
    app.run(debug=True)