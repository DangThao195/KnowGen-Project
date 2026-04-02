import sys
import os
import logging

# Set up tracing for LangChain to output to terminal
try:
    from langchain.globals import set_debug, set_verbose
    set_debug(True)
    set_verbose(True)
except ImportError:
    pass

# Add the backend directory to sys.path to resolve 'app.x' imports
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from app.ingestion.etl import run_etl_pipeline
from app.agents.supervisor_agent import supervisor_node

# Setup basic logging to see traces
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*50)
    print("=== 1. STARTING ETL PIPELINE ===")
    print("="*50)
    sample_file = r"C:\Users\huuda\Downloads\a.thuvienvatly.com.2160e.47524.pdf"
    
    faiss_index_path = os.path.join(os.getcwd(), "vector_store", "faiss_index", "index.faiss")
    if os.path.exists(faiss_index_path):
        print(f"[*] FAISS index already exists at {faiss_index_path}.\n[*] Skipping ETL pipeline to save time.")
    else:
        try:
            run_etl_pipeline(file_paths=[sample_file])
        except ModuleNotFoundError as e:
            print(f"\n[ERROR] Missing Module: {e}")
            print("Please run `pip install langchain_community` (or other missing modules) to fix this.\n")
        except Exception as e:
            print(f"\n[ERROR] ETL Pipeline Error: {e}\n")

    print("\n" + "="*50)
    print("=== 2. TESTING SUPERVISOR NODE ===")
    print("="*50)
    
    state = {"user_query": "Sóng là gì? Giải thích cho tôi"}
    
    try:
        # Trace output of the LLM going off
        print(">> Running supervisor_node(state)...")
        updated_fields = supervisor_node(state)
        
        # Merge updates back into the state graph
        state.update(updated_fields)
        
        print("\n" + "-"*40)
        print("--- FINAL STATE ATTRIBUTES ---")
        print("-"*40)
        for key, value in state.items():
            print(f"\n>>> {key.upper()}:")
            print(value)
            
    except Exception as e:
        print(f"\n[ERROR] Supervisor Node Error: {e}\n")

if __name__ == "__main__":
    main()
