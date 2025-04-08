from flask import Flask, request, jsonify

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from flask import Flask, request, jsonify
import os
import logging
import time
import google.generativeai as genai
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# --- Gemini Model Setup ---
model = None
MODEL_NAME = 'models/gemini-1.5-flash-latest'
MAX_TOKENS = 300 # Max tokens for Gemini response
TEMPERATURE = 0.7 # Temperature for Gemini response

try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in environment variables.")

    genai.configure(api_key=api_key)
    logging.info(f"Initializing Gemini model: {MODEL_NAME}")

    model = genai.GenerativeModel(model_name=MODEL_NAME)

    # --- Optional Startup Test ---
    logging.info("Performing a quick test generation...")
    test_prompt = "Say ok if you are working."
    test_response = model.generate_content(
        test_prompt,
        generation_config=genai.types.GenerationConfig(candidate_count=1)
    )
    if test_response.candidates:
        logging.info("Gemini model initialized successfully.")
        logging.info(f"Sample response: {test_response.text[:80]}...")
    else:
        logging.warning("Model initialized, but test response was empty.")
        logging.warning(f"Test Prompt Feedback: {getattr(test_response, 'prompt_feedback', 'N/A')}")
    # --- End Startup Test ---

except Exception as e:
    logging.error(f"Model initialization failed: {e}")
    model = None

# --- Static Study Material Resource Dictionary ---
# Customize this with relevant topics and high-quality resource links/names
study_resources = {
    "biology": [
        "Khan Academy Biology (https://www.khanacademy.org/science/biology)",
        "Nature Biology Subject Page (https://www.nature.com/subjects/biology)"
    ],
    "photosynthesis": [
        "Khan Academy - Photosynthesis (https://www.khanacademy.org/science/biology/photosynthesis-in-plants)",
        "Britannica - Photosynthesis (https://www.britannica.com/science/photosynthesis)"
    ],
    "quantum physics": [
        "Stanford Encyclopedia of Philosophy - Quantum Mechanics (https://plato.stanford.edu/entries/qm/)",
        "Khan Academy - Quantum Physics (https://www.khanacademy.org/science/physics/quantum-physics)"
    ],
    "calculus": [
        "Khan Academy Calculus 1 (https://www.khanacademy.org/math/calculus-1)",
        "MIT OCW Single Variable Calculus (https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/)"
    ],
    "python": [
        "Official Python Tutorial (https://docs.python.org/3/tutorial/)",
        "Real Python Website (https://realpython.com/)"
    ],
    "jee": [ # Joint Entrance Examination (India)
        "Embibe JEE Study Material (https://www.embibe.com/exams/jee-main-study-material/)",
        "Khan Academy JEE Prep (https://www.khanacademy.org/test-prep/jee)"
    ]
    # Add more topics relevant to your hackathon needs
}

def get_resources_from_query(query: str) -> list:
    """
    Finds resource links based on keywords in the user query.
    Uses simple substring matching against the study_resources dictionary keys.
    """
    matched_links = []
    if not query: # Handle empty query case
        return matched_links

    query_lower = query.lower()
    logging.info(f"Checking for resource keywords in: '{query_lower}'")
    found_keywords = set() # Keep track to avoid duplicate lookups if keywords overlap

    for keyword, links in study_resources.items():
        # Simple substring check - might match parts of words
        # For better accuracy later, consider word boundary checks (regex) or NLP techniques
        if keyword in query_lower and keyword not in found_keywords:
            logging.info(f"Keyword '{keyword}' found.")
            matched_links.extend(links)
            found_keywords.add(keyword)

    logging.info(f"Found {len(matched_links)} resources for query.")
    return matched_links

# --- Webhook Endpoint ---
@app.route('/ask', methods=['POST'])
def ask():
    """Handles Dialogflow requests, gets Gemini response, appends resources."""
    if model is None:
        logging.error("Model is not initialized, cannot process request.")
        return jsonify({"fulfillmentText": "Sorry, the AI model connection is down."})

    try:
        req_data = request.get_json(silent=True)
        if not req_data:
            logging.error("Invalid or empty JSON received in request.")
            return jsonify({"fulfillmentText": "Error: Invalid request received."}), 400

        # Extract user message (handle potential missing keys gracefully)
        user_message = req_data.get("queryResult", {}).get("queryText")
        if not user_message:
            logging.warning("No 'queryResult.queryText' found in request.")
            # Decide how to handle this - maybe return an error or a default message
            return jsonify({"fulfillmentText": "I didn't receive a message to process."})

        logging.info(f"Received user message: {user_message}")

        # --- Step 1: Generate content from Gemini ---
        gemini_reply = "Sorry, I couldn't process that request using the AI model." # Default reply
        try:
            logging.info(f"Sending query to Gemini AI model '{MODEL_NAME}'...")
            start_time = time.time()
            response = model.generate_content(
                user_message, # Send the user's original query
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )
            )
            duration = (time.time() - start_time) * 1000
            logging.info(f"Gemini response time: {duration:.2f} ms")

            # Process Gemini response
            if not response.candidates:
                gemini_reply = "I couldn't generate a response, possibly due to content restrictions."
                feedback = getattr(response, 'prompt_feedback', None)
                if feedback and getattr(feedback, 'block_reason', None):
                    gemini_reply += f" (Reason: {feedback.block_reason.name})"
                logging.warning("Empty response candidates from Gemini.")
            else:
                try:
                    gemini_reply = response.text.strip() # Get the text and remove leading/trailing whitespace
                    logging.info(f"Gemini response: {gemini_reply[:150]}...")
                except Exception as extract_e:
                    logging.error(f"Error extracting text from Gemini response: {extract_e}")
                    gemini_reply = "Sorry, there was an issue reading the AI's response."

        except Exception as gemini_e:
            logging.error(f"Error calling Gemini API: {gemini_e}")
            # Keep the default error message or customize
            gemini_reply = "Sorry, there was a technical problem contacting the AI assistant."

        # --- Step 2: Find and Append Study Resources ---
        final_reply = gemini_reply # Start with the response from Gemini (or error message)
        try:
            resources = get_resources_from_query(user_message)
            if resources:
                resources_text = "\n\n📚 **Here are some potentially helpful resources:**\n"
                for link in resources:
                    resources_text += f"- {link}\n" # Add each link as a bullet point

                final_reply += resources_text # Append the formatted resources
                logging.info("Appended study resources to the response.")
            else:
                logging.info("No relevant study resources found for this query.")
        except Exception as resource_e:
            # Log error but don't necessarily overwrite the Gemini reply
            logging.error(f"Error occurred during resource lookup: {resource_e}")


        # --- Step 3: Return the combined response ---
        fulfillment_response = {"fulfillmentText": final_reply}
        logging.info(f"Sending final response to Dialogflow (first 200 chars): {final_reply[:200]}...")
        return jsonify(fulfillment_response)

    except Exception as e:
        # Catch-all for unexpected errors in the main /ask route logic
        logging.error(f"Unexpected error in /ask route: {e}", exc_info=True)
        return jsonify({"fulfillmentText": "An unexpected server error occurred."})

# --- App Runner ---
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port) # Keep debug=True for hackathon development
