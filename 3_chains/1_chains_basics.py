from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model='glm4')

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes.")
    ]
)

chain = prompt_template | model 

result = chain.invoke({"topic": "lawyers", "joke_count": 3})

print(result.content)

print("\n\n---other way---")

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"topic": "律师", "joke_count": 3})

print(result)