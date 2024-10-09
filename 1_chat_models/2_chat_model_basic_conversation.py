from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

model = ChatOllama(model='llama3.1')

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

result = model.invoke(messages)

print(f"Anseer from AI: {result.content}")


# messages = [
#     SystemMessage(content="Solve the following math problems"),
#     HumanMessage(content="What is 81 divided by 9?"),
#     AIMessage(content="81 divided by 9 is 9"),
#     HumanMessage(content="What is 10 tims 5?"),
# ]

# result = model.invoke(messages)

# print(f"full result: {result}")
# print(f"Anseer from AI: {result.content}")