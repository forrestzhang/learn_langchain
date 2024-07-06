from langchain_community.chat_models import ChatOllama


model = ChatOllama(model='llama3')

# result = model.invoke("What is 81 divided by 9?")
result = model.invoke("81除以9是几?")

print(f"Full result:\n{result}")

print(f"Content:\n{result.content}")