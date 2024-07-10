from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel  
from langchain_community.chat_models import ChatOllama


model = ChatOllama(model='llama3')

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main features the product {product_name}.")
    ]
)


def anayze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "Give these features:{features}, list the pros of these features.")
        ]
    )
    return pros_template.format_prompt(features=features)


def anayze_cons(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "Give these features:{features}, list the cons of these features.")
        ]
    )
    return pros_template.format_prompt(features=features)


def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

pros_branch = (
    RunnableLambda(lambda x: anayze_pros(x)) | model | StrOutputParser()
)

cons_branch = (
    RunnableLambda(lambda x: anayze_cons(x)) | model | StrOutputParser()
)

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"prons": pros_branch, "cons": cons_branch})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["prons"], x["branches"]["cons"]))
)

result = chain.invoke({"product_name": "iPhone 13 Pro"})

print(result)