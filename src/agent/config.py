"""LLm andother configs"""
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from src.models.model import model

load_dotenv()

llm = init_chat_model(model=model.small_model)

NOT_COVERED = "This is not covered in your course material."
MAX_RETRIES = 2
MAX_FIXES = 1
NUM_RESULTS = 6
CHECKPOINT_DB = "checkpoints.db"
STUDY_GROUP = "study-group@example.com"    
OUTBOX_FILE = "outbox/shared_plans.jsonl" 
MAX_REVIEWS = 1             
