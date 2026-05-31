import os
import json
import google.generativeai as genai

def load_api_key():
    key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if key:
        return key
    
    curr_dir = os.getcwd()
    for _ in range(5):
        env_path = os.path.join(curr_dir, '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if 'GEMINI' in line or 'GOOGLE' in line or 'API_KEY' in line:
                        parts = line.strip().split('=')
                        if len(parts) == 2:
                            k, v = parts[0].strip(), parts[1].strip()
                            if k in ['GEMINI_API_KEY', 'GOOGLE_API_KEY']:
                                return v.strip("'\"")
        parent = os.path.dirname(curr_dir)
        if parent == curr_dir:
            break
        curr_dir = parent
    return None

def test_other_models():
    api_key = load_api_key()
    if not api_key:
        print("No API Key found.")
        return
        
    genai.configure(api_key=api_key)
    
    models = [
        'gemini-3.5-flash',
        'models/antigravity-preview-05-2026',
        'models/deep-research-preview-04-2026'
    ]
    
    for m_name in models:
        print(f"\nTrying {m_name}...")
        try:
            model = genai.GenerativeModel(m_name)
            response = model.generate_content("Hello! Are you online? Respond with 'YES'.")
            print(f"-> SUCCESS with {m_name}!")
            print("Response:", response.text.strip())
            break
        except Exception as e:
            print(f"-> FAILED with {m_name}: {e}")

if __name__ == '__main__':
    test_other_models()
