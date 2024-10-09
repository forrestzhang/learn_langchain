from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


model = ChatOllama(model='llama3')

# Part 1, Prompt with one placeholder
template = "Tell me a joke about {topic}."

prompt_template = ChatPromptTemplate.from_template(template=template)

print("---------------Prompt from Template--------------")
prompt = prompt_template.invoke({"topic": "chicken"})
print(prompt)


# Part 2, Prompt with Multiple Placeholders
template_multiple =""" You are a helpful AI assistant.
Human: Tell me a {abjective} story about a {animal}.
Assistant:
"""

prompt_template_multiple = ChatPromptTemplate.from_template(template=template_multiple)
prompt = prompt_template_multiple.invoke({"abjective": "funny", "animal": "panda"})

print("---------------Prompt with Multiple Placeholders--------------")
print(prompt)

# Part 3, Prompt with System and Human Message (Using Tuples)
messages =[
("system","You are a comedian who tells jokes about {topic}."),
("human","Tell me {joke_count} jokes.")
]
prompt_template_messages = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template_messages.invoke({"topic": "lawyers", "joke_count": 3})
print("---------------Prompt with System and Human Message (Tuple)--------------")
print(prompt)


# Warning This DO NOT WORK
messages =[
    ("system","You are a comedian who tells jokes about {topic}."),
    HumanMessage("Tell me {joke_count} jokes.")
]
prompt_template_messages = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template_messages.invoke({"topic": "lawyers", "joke_count": 3})
print("---------------Prompt with System and Human Message (Tuple)--------------")
print(prompt)
