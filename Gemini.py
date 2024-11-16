from google.cloud import dialogflow_v2 as dialogflow

# Initialize the client
project_id = "your-project-id"
session_id = "unique-session-id"
language_code = "en"

# Create a session client
client = dialogflow.SessionsClient()

def detect_intent_texts(project_id, session_id, texts, language_code):
    session = client.session_path(project_id, session_id)
    print(f"Session path: {session}\n")
    
    for text in texts:
        text_input = dialogflow.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.QueryInput(text=text_input)
        
        response = client.detect_intent(request={"session": session, "query_input": query_input})
        
        print("Query text:", response.query_result.query_text)
        print("Detected intent:", response.query_result.intent.display_name)
        print("Fulfillment text:", response.query_result.fulfillment_text)
        print("\n")

# Chatbot loop
print("Chatbot initialized. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    # Send user input to Gemini/Dialogflow
    detect_intent_texts(project_id, session_id, [user_input], language_code)
