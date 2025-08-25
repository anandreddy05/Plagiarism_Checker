import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Just use the API key
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print("Connection successful!")
print(pc.list_indexes())