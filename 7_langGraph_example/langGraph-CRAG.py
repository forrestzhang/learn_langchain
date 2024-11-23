import marimo

__generated_with = "0.9.20"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ```bash
        pip install  nomic[local] langchain-nomic
        ```
        """
    )
    return


@app.cell
def __():
    from langchain_chroma import Chroma
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    # from langchain_community.vectorstores import SKLearnVectorStore
    # from langchain_nomic.embeddings import NomicEmbeddings  # local
    # from langchain_openai import OpenAIEmbeddings
    from langchain_community.document_loaders import WebBaseLoader

    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    return (
        ChatOllama,
        Chroma,
        JsonOutputParser,
        OllamaEmbeddings,
        PromptTemplate,
        RecursiveCharacterTextSplitter,
        WebBaseLoader,
    )


@app.cell
def __():
    llm_model_name = 'qwen2.5:14b'
    embedding_model_name = 'mxbai-embed-large'
    ollama_base_url = 'http://10.127.127.4:11434'
    return embedding_model_name, llm_model_name, ollama_base_url


@app.cell
def __(
    ChatOllama,
    OllamaEmbeddings,
    embedding_model_name,
    llm_model_name,
    ollama_base_url,
):
    embedding = OllamaEmbeddings(model=embedding_model_name, 
                                base_url=ollama_base_url)

    llm_json = ChatOllama(model=llm_model_name, format='json', temperature=0, base_url=ollama_base_url)
    return embedding, llm_json


@app.cell
def __(mo):
    mo.md(r"""## Create Index""")
    return


@app.cell
def __(RecursiveCharacterTextSplitter, WebBaseLoader):
    # List of URLs to load documents from
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    # Load documents from the URLs
    web_docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in web_docs for item in sublist]

    # Initialize a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )

    # Split the documents into chunks
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits, docs_list, text_splitter, urls, web_docs


@app.cell
def __(Chroma, doc_splits, embedding):
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name='rage-chroma',
        embedding=embedding
    )
    return (vectorstore,)


@app.cell
def __(vectorstore):
    retriever = vectorstore.as_retriever()
    return (retriever,)


@app.cell
def __(retriever):
    retriever.invoke('Agent memory')
    return


@app.cell
def __(mo):
    mo.md(r"""## Define Tools""")
    return


@app.cell
def __(PromptTemplate):
    prompt = PromptTemplate(
        template="""You are a teacher grading a quiz. You will be given: 
        1/ a QUESTION
        2/ A FACT provided by the student

        You are grading RELEVANCE RECALL:
        A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
        A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION. 
        1 is the highest (best) score. 0 is the lowest score you can give. 

        Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 

        Avoid simply stating the correct answer at the outset.

        Question: {question} \n
        Fact: \n\n {documents} \n\n

        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """,
        input_variables=["question", "documents"],
    )


    return (prompt,)


@app.cell
def __(JsonOutputParser, llm_json, prompt, retriever):
    retrieval_grader = prompt | llm_json | JsonOutputParser()
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    print(retrieval_grader.invoke({"question": question, "documents": doc_txt}))
    return doc_txt, docs, question, retrieval_grader


@app.cell
def __(
    ChatOllama,
    PromptTemplate,
    docs,
    llm_model_name,
    ollama_base_url,
    question,
):
    ### Generate

    from langchain_core.output_parsers import StrOutputParser

    # Prompt
    prompt_2 = PromptTemplate(
        template="""You are an assistant for question-answering tasks. 
        
        Use the following documents to answer the question. 
        
        If you don't know the answer, just say that you don't know. 
        
        Use three sentences maximum and keep the answer concise:
        Question: {question} 
        Documents: {documents} 
        Answer: 
        """,
        input_variables=["question", "documents"],
    )

    # LLM
    llm_plain_text = ChatOllama(model=llm_model_name, temperature=0, base_url=ollama_base_url)

    # Chain
    rag_chain = prompt_2 | llm_plain_text | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"documents": docs, "question": question})
    print(generation)
    return StrOutputParser, generation, llm_plain_text, prompt_2, rag_chain


@app.cell
def __():
    from langchain_community.tools import DuckDuckGoSearchResults

    web_search_tool = DuckDuckGoSearchResults(output_format="list")
    return DuckDuckGoSearchResults, web_search_tool


@app.cell
def __():
    from typing import List
    from typing_extensions import TypedDict
    # from IPython.display import Image, display
    from langchain.schema import Document
    from langgraph.graph import START, END, StateGraph
    return Document, END, List, START, StateGraph, TypedDict


@app.cell
def __(
    Document,
    END,
    List,
    START,
    StateGraph,
    TypedDict,
    rag_chain,
    retrieval_grader,
    retriever,
    web_search_tool,
):
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            search: whether to add search
            documents: list of documents
        """

        question: str
        generation: str
        search: str
        documents: List[str]
        steps: List[str]


    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        question = state["question"]
        documents = retriever.invoke(question)
        steps = state["steps"]
        steps.append("retrieve_documents")
        return {"documents": documents, "question": question, "steps": steps}


    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """

        question = state["question"]
        documents = state["documents"]
        generation = rag_chain.invoke({"documents": documents, "question": question})
        steps = state["steps"]
        steps.append("generate_answer")
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "steps": steps,
        }


    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        question = state["question"]
        documents = state["documents"]
        steps = state["steps"]
        steps.append("grade_document_retrieval")
        filtered_docs = []
        search = "No"
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "documents": d.page_content}
            )
            grade = score["score"]
            if grade == "yes":
                filtered_docs.append(d)
            else:
                search = "Yes"
                continue
        return {
            "documents": filtered_docs,
            "question": question,
            "search": search,
            "steps": steps,
        }


    def web_search(state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        question = state["question"]
        documents = state.get("documents", [])
        steps = state["steps"]
        steps.append("web_search")
        web_results = web_search_tool.invoke(question)
        documents.extend(
            [
                Document(page_content=d["content"], metadata={"url": d["url"]})
                for d in web_results
            ]
        )
        return {"documents": documents, "question": question, "steps": steps}

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        search = state["search"]
        if search == "Yes":
            return "search"
        else:
            return "generate"


    # Graph
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("web_search", web_search)  # web search

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    custom_graph = workflow.compile()
    return (
        GraphState,
        custom_graph,
        decide_to_generate,
        generate,
        grade_documents,
        retrieve,
        web_search,
        workflow,
    )


@app.cell
def __(custom_graph, mo):
    mo.image(custom_graph.get_graph(xray=True).draw_mermaid_png())
    return


@app.cell
def __(custom_graph):
    import uuid


    def predict_custom_agent_local_answer(example: dict):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state_dict = custom_graph.invoke(
            {"question": example["input"], "steps": []}, config
        )
        return {"response": state_dict["generation"], "steps": state_dict["steps"]}


    example = {"input": "What are the types of agent memory?"}
    response = predict_custom_agent_local_answer(example)
    response
    return example, predict_custom_agent_local_answer, response, uuid


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
