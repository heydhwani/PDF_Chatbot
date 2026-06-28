from api import client

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Hello World"
)

print(response)