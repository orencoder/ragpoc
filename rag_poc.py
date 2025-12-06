### v3


import logging
import sys
from typing import List

# ייבוא רכיבי LlamaIndex
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, ExactMatchFilter
import qdrant_client

# 1. הגדרות המודלים (עובדים לוקאלית על המק)
print("Loading models...")

# מודל Embedding שתומך בעברית (קטן ויעיל ל-POC)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# מודל השפה שרץ ב-Ollama
Settings.llm = Ollama(model="qwen2.5:14b", request_timeout=120.0)

# 2. חיבור ל-Qdrant
client = qdrant_client.QdrantClient(location="http://localhost:6333")
vector_store = QdrantVectorStore(client=client, collection_name="security_poc")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 3. יצירת דאטה מדומה (Simulated Data)
# נדמה שני מסמכים: אחד ציבורי ואחד סודי (HR)
documents = [
    Document(
        text="מדיניות החופשות הכללית: כל עובד זכאי ל-12 ימי חופשה בשנה. יש לתאם מול המנהל.",
        metadata={"filename": "general_policy.pdf", "allowed_groups": ["all_employees", "hr_managers"]},
    ),
    Document(
        text="דוח בונוסים סודי 2024: יוסי כהן יקבל בונוס של 50,000 שח. דני לוי יקבל 10,000 שח.",
        metadata={"filename": "secret_bonuses.pdf", "allowed_groups": ["hr_managers"]}, # שים לב: רק ל-HR מותר
    ),
]

# אינדוקס המסמכים לתוך Qdrant
print("Indexing documents...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

# 4. פונקציית השאילתה המאובטחת
def ask_securely(question: str, user_groups: List[str], user_name: str):
    print(f"\n--- שואל בתור: {user_name} (קבוצות: {user_groups}) ---")
    
    # בניית הפילטר: תביא מסמכים רק אם הקבוצה של המשתמש מופיעה ב-allowed_groups
    # הערה: בלוגיקה אמיתית משתמשים ב-Filter מורכב יותר, כאן לצורך הפשטות נשתמש בפילטר בסיסי
    # ב-POC זה, אנחנו ניצור פילטרים עבור כל קבוצה שיש למשתמש
    
    filters_list = [
        ExactMatchFilter(key="allowed_groups", value=group) for group in user_groups
    ]
    
    # שימוש ב-OR: מספיק שאחת הקבוצות תתאים
    query_filters = MetadataFilters(filters=filters_list, condition="or")

    # יצירת מנוע חיפוש עם הפילטרים
    query_engine = index.as_query_engine(
        filters=query_filters,
        similarity_top_k=3
    )
    
    response = query_engine.query(question)
    print(f"שאלה: {question}")
    print(f"תשובה: {response}")

# --- שלב ההרצה ---

# תרחיש א': משתמש רגיל ("דני") מנסה לגלות כמה בונוס קיבל יוסי
ask_securely(
    question="כמה בונוס קיבל יוסי כהן?", 
    user_groups=["all_employees"], 
    user_name="Dani (Junior Employee)"
)

# תרחיש ב': מנהל HR ("רינה") שואלת את אותו הדבר
ask_securely(
    question="כמה בונוס קיבל יוסי כהן?", 
    user_groups=["hr_managers"], 
    user_name="Rina (HR Manager)"
)