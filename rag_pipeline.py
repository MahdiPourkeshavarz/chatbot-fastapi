# rag_pipeline.py - FINAL VERSION WITH FARSI-SAFE DETECTION

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import chromadb
from sentence_transformers import SentenceTransformer
import json
import re
import unicodedata
from typing import List, Dict, Any

# --- CONFIGURATION CONSTANTS (MUST MATCH YOUR SETUP) ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DB_PATH = "dishdash_vector_db" # Path to the unzipped ChromaDB folder!

# Weighted Score Constants
MINIMUM_VOTES_M = 10.0
OVERALL_MEAN_C = 3.5

# --- AMENITY MAPPING (Your custom Farsi/English filter) ---
AMENITY_MAP = {
    "cafe": "cafe", "coffee": "cafe", "کافه": "cafe", "قهوه": "cafe",
    "کافی شاپ": "cafe", "شیک": "cafe", "دسر": "cafe", "شیرینی": "cafe",
    "فالوده": "cafe", "بستنی ایرانی": "cafe", "باقلوا": "cafe",
    "زولبیا و بامیه": "cafe", "سوهان": "cafe", "گز": "cafe",
    "نبات": "cafe", "نخودچی": "cafe", "قرابیه": "cafe",
    "شیرینی برنجی": "cafe", "شیرینی گوش‌فیل": "cafe", "دلمه برگ‌مو": "cafe",
    "فرنی": "cafe", "کلوچه": "cafe", "شربت": "cafe",
    "ماست و خیار": "cafe", "دوغ": "cafe", "سکنجبین": "cafe",
    "شیرینی نخودچی": "cafe",
    "fast food": "fast_food", "فست فود": "fast_food",
    "پیتزا": "fast_food", "برگر": "fast_food", "ساندویچ": "fast_food",
    "سوخاری": "fast_food", "دونر": "fast_food", "پاستا": "fast_food",
    "همبرگر": "fast_food", "مرغ سوخاری": "fast_food", "کریسپی": "fast_food",
    "هات داگ": "fast_food",
    "restaurant": "restaurant", "رستوران": "restaurant", "غذا": "restaurant",
    "ناهار": "restaurant", "شام": "restaurant", "ایرانی": "restaurant",
    "سنتی": "restaurant", "دیزی": "restaurant",
    "کباب": "restaurant", "چلوکباب": "restaurant", "کباب بختیاری": "restaurant",
    "جوجه‌کباب": "restaurant", "کباب کوبیده": "restaurant", "کباب سلطانی": "restaurant",
    "کباب چنجه": "restaurant", "شیشلیک": "restaurant", "کباب برگ": "restaurant",
    "کباب شیشلیک": "restaurant", "کباب تابه‌ای": "restaurant", "کباب ترش": "restaurant",
    "کباب ترش گیلکی": "restaurant", "جیگر": "restaurant",
    "سبزی پلو": "restaurant", "زرشک‌پلو": "restaurant", "باقلاپلو": "restaurant",
    "خورش قیمه": "restaurant", "شیرین‌پلو": "restaurant", "عدس‌پلو": "restaurant",
    "آلبالوپلو": "restaurant", "دمپختک": "restaurant", "رشته‌پلو": "restaurant",
    "کلم‌پلو": "restaurant", "لوبیاپلو": "restaurant", "نخودپلو": "restaurant",
    "باقالی‌پلو با ماهیچه": "restaurant", "مرصع‌پلو": "restaurant", "زرشک‌پلو با مرغ": "restaurant",
    "آبگوشت": "restaurant", "فسنجان": "restaurant", "قرمه‌سبزی": "restaurant",
    "میرزاقاسمی": "restaurant", "آش": "restaurant", "اشکنه": "restaurant",
    "هلیم": "restaurant", "قلیه ماهی": "restaurant", "قلیه میگو": "restaurant",
    "عدسی": "restaurant", "کله پاچه": "restaurant", "کوفته": "restaurant",
    "یتیمچه": "restaurant", "ترخینه": "restaurant", "ته‌چین": "restaurant",
    "کوفته تبریزی": "restaurant", "آش رشته": "restaurant", "آش دوغ": "restaurant",
    "آش شله قلمکار": "restaurant", "خورش بادمجان": "restaurant", "خورش کرفس": "restaurant",
    "خورش بامیه": "restaurant", "باقالی قاتق": "restaurant", "سیرابی": "restaurant",
    "شله مشهدی": "restaurant",
    "دلمه": "restaurant", "بریان": "restaurant", "ترشی": "restaurant",
    "حلوا": "restaurant", "خاگینه": "restaurant", "سمبوسه": "restaurant",
    "شله زرد": "restaurant", "شیربرنج": "restaurant", "کاچی": "restaurant",
    "کله‌جوش": "restaurant", "کوکو": "restaurant", "کتلت": "restaurant",
    "ماهی شکم‌پر": "restaurant", "ماهی قزل‌آلا": "restaurant", "ماهی کبابی": "restaurant",
    "نان سنگک": "restaurant", "نان بربری": "restaurant", "نان لواش": "restaurant",
    "نان تافتون": "restaurant", "بورک": "restaurant",
}

def get_amenity_filter(query):
    lower_query = query.lower()
    for keyword, amenity in AMENITY_MAP.items():
        if keyword in lower_query:
            return {"tags_amenity": amenity}
    return None

# --- FARSI-SAFE NORMALIZATION AND DETECTION HELPER FUNCTIONS ---

def normalize_farsi(text: str) -> str:
    """Converts common Arabic characters to their standard Farsi equivalents and normalizes."""
    if not text:
        return ""
    # Standardize KAF and YEH for consistency
    text = text.replace('ك', 'ک')  # Arabic Kaf (ك) to Farsi Kaf (ک)
    text = text.replace('ي', 'ی')  # Arabic Yeh (ي) to Farsi Yeh (ی)
    # Apply NFKC normalization
    return unicodedata.normalize('NFKC', text)

def detect_mentioned_places_farsi_safe(llm_response: str, context_to_use: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Guarantees detection of place names by normalizing Farsi characters and using Regex
    to scan the LLM's response against the top retrieved names.
    """

    name_id_lookup = {}
    place_names_normalized = []

    # 1. Normalize Names and Create Lookup Table
    for place in context_to_use:
        original_name = place.get('name')
        place_id = str(place.get('_id'))

        if original_name and place_id:
            normalized_name = normalize_farsi(original_name)
            name_id_lookup[normalized_name] = {"id": place_id, "original_name": original_name}

            # Use the escaped normalized name for the regex pattern
            place_names_normalized.append(re.escape(normalized_name))

    if not place_names_normalized:
        return []

    # 2. Normalize the LLM Response
    normalized_llm_response = normalize_farsi(llm_response)

    # 3. Dynamically create the Regex pattern: (Name 1|Name 2|...)
    pattern = r'(?:' + '|'.join(place_names_normalized) + r')'

    # 4. Find all unique matches in the normalized LLM response
    detected_names_normalized = set(re.findall(pattern, normalized_llm_response))

    # 5. Build the Final Structured List
    final_structured_recommendations = []

    for detected_name in detected_names_normalized:
        # Find the original data using the normalized name from the lookup table
        if detected_name in name_id_lookup:
             data = name_id_lookup[detected_name]
             final_structured_recommendations.append({
                "id": data["id"],
                "name": data["original_name"] # Return the original Farsi name to the user
            })
             # Ensure the list only contains unique, detected places by removing from lookup
             del name_id_lookup[detected_name]

    return final_structured_recommendations


# --- RAG Pipeline Core Functions (load_rag_pipeline and ask_dishdash) ---

def load_rag_pipeline():
    """Loads all models and the database only once for the FastAPI server."""
    print("🚀 Initializing RAG Pipeline (Loading Models and DB)...")

    # 1. Load LLM and Tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 2. Load Embedding Model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 3. Load Persistent ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name="places")

    print(f"✅ RAG Pipeline ready! DB items: {collection.count()}")

    return model, tokenizer, embedding_model, collection

def ask_dishdash(user_query, model, tokenizer, embedding_model, collection):
    """
    Executes the full RAG pipeline: Filter -> Retrieve -> Rank -> Generate -> Detect.
    """

    amenity_filter = get_amenity_filter(user_query)
    query_embedding = embedding_model.encode(user_query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10,
        include=['metadatas'],
        where=amenity_filter
    )

    context_places = results.get('metadatas', [[]])[0]

    if not context_places:
        return { "llm_response": "I couldn't find any places matching that specific request.", "recommended_places": [] }

    # Weighted Score Re-ranking
    weighted_places = []
    for p in context_places:
        R = p.get('averageRating')
        v = p.get('ratingCount')
        if R is not None and v is not None:
            try:
                R = float(R)
                v = float(v)
                weighted_score = ( (v * R) + (MINIMUM_VOTES_M * OVERALL_MEAN_C) ) / (v + MINIMUM_VOTES_M)
                p['weighted_score'] = weighted_score
                weighted_places.append(p)
            except (ValueError, TypeError): pass

    sorted_places = sorted(weighted_places, key=lambda x: x.get('weighted_score', 0), reverse=True)[:5]
    context_to_use = sorted_places if sorted_places else context_places[:5]

    # Building the Strict Prompt (Forces LLM to use the names in a list)
    prompt_template = f"""
    <s>[INST] **IMPORTANT**: You are DishDash, a friendly and energetic food finder bot.
    **TONE**: Your answer MUST be fun and friendly, like a knowledgeable friend.
    **CONSTRAINT**: Your ONLY task is to create a numbered list (1, 2, 3...) of the top {len(context_to_use)} places based on the user's request.
    **OUTPUT FORMAT**: Each item in the list MUST contain ONLY the place name. DO NOT include ratings, phone numbers, coordinates, or any descriptive text.

    **Database Search Results (Top {len(context_to_use)} Weighted Places):**
    {json.dumps(context_to_use, indent=2, ensure_ascii=False)}

    **User's Question:**
    {user_query}

    [/INST]
    Helpful Answer (Start immediately with your fun response, followed by the numbered list of names ONLY):
    """

    # LLM Generation
    inputs = tokenizer(prompt_template, return_tensors="pt", padding=True, truncation=True).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    try:
        llm_answer = response_text.split("Helpful Answer (Start immediately with your fun response, followed by the numbered list of names ONLY):")[1].split("[/INST]")[-1].strip()
    except IndexError:
        llm_answer = response_text.split("[/INST]")[-1].strip()

    # --- POST-GENERATION PROCESSING (PGP) ---
    # Guaranteed detection of names actually mentioned in the LLM's text:
    final_structured_recommendations = detect_mentioned_places_farsi_safe(llm_answer, context_to_use)

    # --- FINAL RETURN STRUCTURE ---
    return {
        "llm_response": llm_answer,
        "recommended_places": final_structured_recommendations
    }