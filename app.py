from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import json
from PIL import Image

app = Flask(__name__)


API_KEY = "AIzaSyCd4Uhs7sCiGWOZsqClmce0d9O4z5ZP79M"

# Configure the AI model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

@app.route('/')
def home():
    return render_template('index.html')

# --- FEATURE 1: AGENTIC CLARIFICATION LOOP ---
@app.route('/clarify', methods=['POST'])
def clarify_prompt():
    data = request.json
    weak_prompt = data.get('prompt')
    
    system_instruction = f"""
    You are an expert AI Assistant. The user wants to write a prompt: "{weak_prompt}".
    This prompt is too vague. Ask exactly 2 short, multiple-choice questions to figure out their true intent (e.g., audience, tone, format).
    Return ONLY a valid JSON object with the key "questions" containing a list of 2 strings.
    """
    try:
        response = model.generate_content(
            system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )
        return jsonify(json.loads(response.text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- FEATURE 2: OPTIMIZER & ECO-PROMPTING ---
@app.route('/optimize', methods=['POST'])
def optimize_prompt():
    data = request.json
    weak_prompt = data.get('prompt')
    answers = data.get('answers', 'None provided')
    target_model = data.get('target_model')

    system_instruction = f"""
    You are an expert AI Prompt Engineer. 
    Original weak prompt: "{weak_prompt}"
    User Context/Clarifications: "{answers}"
    Target Model: {target_model}
    
    1. Score the original prompt out of 100.
    2. Write an analysis of what was missing.
    3. Write a 'standard_prompt': highly detailed, using best practices for {target_model}. If the user context contains a "Previous Failure Insight", ensure you fix that specific flaw in the new prompt.
    4. Write an 'eco_prompt': compress the standard prompt to use 30% fewer words while keeping the exact meaning (abbreviations, no filler).
    
    Return ONLY a valid JSON object with keys: "score", "analysis", "standard_prompt", "eco_prompt"
    """
    try:
        response = model.generate_content(
            system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )
        result = json.loads(response.text)
        
        # Calculate rough token estimates (1 token ~= 4 chars)
        std_tokens = len(result.get('standard_prompt', '')) // 4
        eco_tokens = len(result.get('eco_prompt', '')) // 4
        savings = 0 if std_tokens == 0 else int(((std_tokens - eco_tokens) / std_tokens) * 100)
        
        result['token_stats'] = {
            "standard": std_tokens,
            "eco": eco_tokens,
            "savings": savings
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- NEW FEATURE: SELF-CORRECTION LOOP (RLHF) ---
@app.route('/refine', methods=['POST'])
def refine_prompt():
    data = request.json
    weak_prompt = data.get('prompt')
    previous_answers = data.get('answers')
    optimized_prompt = data.get('optimized_prompt')
    user_feedback = data.get('feedback')

    system_instruction = f"""
    You are an expert AI Prompt Engineer debugging a prompt that failed to meet the user's expectations.
    
    [Context]
    Original intent: "{weak_prompt}"
    User's previous answers: "{previous_answers}"
    The generated prompt we just gave them: "{optimized_prompt}"
    
    [The Problem]
    The user reported the following flaw/feedback regarding the generated prompt: "{user_feedback}"
    
    [Your Task]
    Generate EXACTLY ONE highly specific, targeted clarifying question to ask the user. This question should gather the exact piece of missing information needed to fix the flaw they mentioned.
    Keep the question concise and helpful. 
    
    Return ONLY a valid JSON object with the key "question" containing the string.
    """
    try:
        response = model.generate_content(
            system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )
        return jsonify(json.loads(response.text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- FEATURE 3: REVERSE PROMPT ENGINEERING (TEXT) ---
@app.route('/reverse', methods=['POST'])
def reverse_engineer():
    data = request.json
    target_text = data.get('text')
    
    system_instruction = f"""
    You are an expert prompt reverse-engineer. 
    The user has provided some input text. 
    - If the text looks like a finished piece of writing (an essay, article, or story), write the perfect, detailed prompt that would generate that exact tone, style, and content.
    - If the text is a short, vague request (like "i want a picture to draw" or "write a blog"), expand it into a highly detailed, professional master prompt.
    
    User Input: "{target_text}"
    
    Return ONLY a valid JSON object with the key "extracted_prompt" containing the prompt string.
    """
    try:
        response = model.generate_content(
            system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )
        return jsonify(json.loads(response.text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- FEATURE 4: IMAGE-TO-PROMPT (VISION REVERSE ENGINEERING) ---
@app.route('/reverse-image', methods=['POST'])
def reverse_image_engineer():
    # Check if an image was uploaded in the form data
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # Open the image file via Pillow (PIL)
        img = Image.open(file.stream)
        
        system_instruction = """
        You are an expert AI image prompt engineer. 
        Analyze the provided image in exquisite detail. 
        Write the perfect, highly detailed text-to-image prompt (suitable for Midjourney, DALL-E, or Stable Diffusion) that would generate an image as similar to this one as possible.
        Be sure to include details about the primary subject, background, lighting, camera angle, art style, medium, colors, and overall mood.
        
        Return ONLY a valid JSON object with the key "extracted_prompt" containing the prompt string.
        """
        
        # Pass both the text instructions and the image object to Gemini
        response = model.generate_content(
            [system_instruction, img],
            generation_config={"response_mime_type": "application/json"}
        )
        return jsonify(json.loads(response.text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)