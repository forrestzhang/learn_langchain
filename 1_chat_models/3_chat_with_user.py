from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


model = ChatOllama(model="llama3.1")

chat_history = []

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

while True:
    
    query = input("You: ")
    
    if query.lower() == "exit":
        break
    
    chat_history.append(HumanMessage(content=query))
    
    result = model.invoke(chat_history)
    
    response = result.content
    
    chat_history.append(AIMessage(content=response))
    
    print("AI: ", response)
    
print("--------History----------")
print(chat_history)
print("Goodbye!")