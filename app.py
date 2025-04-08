from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app, origins=os.getenv('ALLOWED_ORIGINS', '*').split(','))

# Constants
MAX_CONTEXT_DEPTH = 3  # Number of previous exchanges to remember
MAX_TOKENS = 1000
TEMPERATURE = 0.7
MODEL_NAME = 'models/gemini-1.5-flash-latest'

# Gemini Setup
model = None
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(MODEL_NAME)
    logging.info(f"Initialized Gemini model: {MODEL_NAME}")
except Exception as e:
    logging.error(f"Gemini initialization failed: {e}")

# Study Resources (Example)
study_resources = {
    "python": ["https://docs.python.org/3/", "https://www.learnpython.org/"],
    "flask": ["https://flask.palletsprojects.com/", "https://realpython.com/flask-by-example/"],
    "javascript": ["https://developer.mozilla.org/en-US/docs/Web/JavaScript", "https://javascript.info/"]
}

def get_resources_from_query(query: str) -> list:
    """Extract relevant study resources based on query"""
    resources = []
    for keyword in study_resources:
        if keyword.lower() in query.lower():
            resources.extend(study_resources[keyword])
    return resources

def validate_context(context):
    """Ensure context has valid structure"""
    if not isinstance(context, list):
        return False
    for item in context:
        if not isinstance(item, dict):
            return False
        if 'role' not in item or 'content' not in item:
            return False
        if item['role'] not in ('user', 'bot'):
            return False
    return True

@app.route('/ask', methods=['POST'])
def ask():
    if not model:
        return jsonify({
            "fulfillmentText": "AI service unavailable",
            "context": []
        }), 503

    try:
        req_data = request.get_json()
        if not req_data:
            return jsonify({"fulfillmentText": "Invalid request", "context": []}), 400

        # Input Validation
        context = req_data.get('context', [])
        current_message = req_data.get('currentMessage', '').strip()

        if not validate_context(context):
            context = []
            logging.warning("Invalid context structure received")

        if not current_message:
            return jsonify({"fulfillmentText": "Please enter a question", "context": context}), 400

        # Context Management
        valid_context = context[-(MAX_CONTEXT_DEPTH * 2):]

        # Build Gemini History
        gemini_history = []
        for exchange in valid_context:
            role = "user" if exchange['role'] == "user" else "model"
            gemini_history.append({"role": role, "parts": [{"text": exchange['content']}]})

        # Add current message
        gemini_history.append({"role": "user", "parts": [{"text": current_message}]})

        # Generate Response
        try:
            response = model.generate_content(
                gemini_history,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )
            )
            bot_response = response.text.strip() if response.candidates else "Couldn't generate response"
        except Exception as e:
            logging.error(f"Gemini API error: {str(e)}")
            bot_response = "Error processing request"

        # Append Study Resources
        try:
            resources = get_resources_from_query(current_message)
            if resources:
                bot_response += "\n\n📚 Resources:\n" + "\n".join([f"- {link}" for link in resources])
        except Exception as resource_e:
            logging.error(f"Resource lookup error: {resource_e}")

        # Update Context
        new_context = valid_context + [
            {"role": "user", "content": current_message},
            {"role": "bot", "content": bot_response}
        ][-(MAX_CONTEXT_DEPTH * 2):]

        return jsonify({
            "fulfillmentText": bot_response,
            "context": new_context
        })

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "fulfillmentText": "An internal error occurred",
            "context": []
        }), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
