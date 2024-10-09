from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_ollama import ChatOllama


model = ChatOllama(model='llama3')

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a help assisant."),
        ("human", "Generate a thank you note for this positive feedback: {feedback}.")
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a help assisant."),
        ("human", "Generate a negative note for this negative feedback: {feedback}.")
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a help assisant."),
        ("human", "Generate a neutral note for this netural feedback: {feedback}.")
    ]
)

escalate_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a help assisant."),
        ("human", "Generate a message to escalate this feedback to a human agent: {feedback}.")
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a help assisant."),
        ("human", "Classify the sentiment of this feedback as positive, negative, netral or escalate: {feedback}.")
    ]
)


branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_template | model | StrOutputParser() # Default branch
    
    
)


classification_chain = (
    classification_template
    | model
    | StrOutputParser()
    
)   

chain = classification_chain | branches

# Good Review
# review = "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad Review
# review = "This product is terrible. It broke after just one use and the quality is very poor."
# Neutral Review
# review = "The product is okay. It works as expected but nothing special."
# Default
review = "I'm not sure about this product. Can you tell me more about this?"

result = chain.invoke({"feedback": review})

print(result)