import os

# 导入所需的库和模块
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_ollama import ChatOllama

# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.realpath(__file__))
# 构建持久化目录路径
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# 初始化嵌入模型
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')

# 初始化Chroma数据库
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# 配置检索器
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "score_threshold": 0.4})

# 初始化语言模型
llm = ChatOllama(model="llama3")

# 定义用于生成独立问题的系统提示
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history,"
    "formulate a stanalone question which can be understuood"
    "without the chat history. Do NOT answer the question, just "
    "reformulat it if needed and other wise return it as is."
)

# 创建用于生成独立问题的提示模板
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)

# 创建具有历史意识的检索器
history_aware_retiever = create_history_aware_retriever(retriever=retriever, llm=llm,
                                                        prompt=contextualize_q_prompt)

# 定义用于回答问题的系统提示
qa_system_prompt = (
    "You are an assistant for quesiton-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that"
    "you don't know. Use three sentence maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# 创建用于回答问题的提示模板
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# 创建用于回答问题的链
question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

# 创建检索增强生成链
rag_chain = create_retrieval_chain(retriever=history_aware_retiever, 
                                   combine_docs_chain=question_answer_chain)

# 定义持续聊天函数
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation" )
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break   
        result = rag_chain.invoke({'chat_history':chat_history, 'input':query})
        print(f"AI: {result['answer']}")

        chat_history.append(HumanMessage(query))
        chat_history.append(SystemMessage(result['answer']))

# 如果作为主模块运行，则启动持续聊天函数
if __name__ == "__main__":
    continual_chat()