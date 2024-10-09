from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories.file import FileChatMessageHistory


model = ChatOllama(model="llama3.1")

chat_history = FileChatMessageHistory("chat_history.json")

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.add_message(system_message)

while True:
    
    query = input("You: ")
    
    if query.lower() == "exit":
        break
    
    chat_history.add_user_message(HumanMessage(content=query))
    
    result = model.invoke(chat_history.messages)
    
    response = result.content
    
    chat_history.add_ai_message(AIMessage(content=response))
    
    print("AI: ", response)
    
print("--------History----------")
print(chat_history)
print("Goodbye!")