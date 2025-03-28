import requests

url = "http://localhost:8000/api/v1/models/meta-llama/Prompt-Guard-86M/jailbreak-score"
headers = {
  "Content-Type": "application/json",
  "X-API-Key": "your-api-key"
}
data = {
  "text": "Sample text to analyze",
  "temperature": 1.0
}

response = requests.post(url, headers=headers, json=data)

print(response.status_code)
print(response.json())