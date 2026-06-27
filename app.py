import os
from dotenv import load_dotenv

load_dotenv()

print(os.getenv("GOOGLE_API_KEY"))

message = create_vectorstore(pdf_path)
print(message)