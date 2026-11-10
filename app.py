import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

GITHUB_URL = "https://github.com/nambak/fullstack-gpt"
CHAT_MODEL = "gpt-5.6-terra"
EMBEDDING_MODEL = "text-embedding-3-small"

st.set_page_config(page_title="DocumentGPT", page_icon="📃")
st.title("📃 DocumentGPT")

with st.sidebar:
    api_key = st.text_input("OpenAI API key", type="password")
    file = st.file_uploader("문서 파일을 올려 주세요 (.txt)", type=["txt"])
    st.markdown(f"[🔗 GitHub 저장소 보기]({GITHUB_URL})")


def reset_chat():
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)


if "messages" not in st.session_state:
    reset_chat()


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for item in st.session_state["messages"]:
        with st.chat_message(item["role"]):
            st.markdown(item["message"])


@st.cache_resource(show_spinner="문서를 임베딩하는 중...")
def embed_file(file, api_key):
    os.makedirs("./.cache/files", exist_ok=True)
    file_path = os.path.join("./.cache/files", os.path.basename(file.name))
    with open(file_path, "wb") as f:
        f.write(file.getvalue())

    loader = TextLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)
    cache_dir = LocalFileStore("./.cache/embeddings/")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
        namespace=EMBEDDING_MODEL,
        key_encoder="sha256",
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 도움이 되는 어시스턴트입니다.
    아래 문서 내용(context)만 사용해서 질문에 답하세요.
    답을 모르면 모른다고 답하고, 지어내지 마세요. 답변은 한국어로 하세요.
    
    {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

if not api_key:
    st.info("왼쪽 사이드바에 OpenAI API 키를 입력해 주세요.")
elif not file:
    reset_chat()
    st.info("왼쪽 사이ㄷ바에서 문서 파일을 올려 주세요.")
else:
    if st.session_state.get("file_id") != file.file_id:
        reset_chat()
        st.session_state["file_id"] = file.file_id

    try:
        retriever = embed_file(file, api_key)
    except Exception as e:
        st.error(f"문서를 처리하지 못했어요. API 키가 맞는지 확인해 주세요.\n\n{e}")
        st.stop()

    llm = ChatOpenAI(model=CHAT_MODEL, openai_api_key=api_key)
    memory = st.session_state["memory"]

    def load_memory(_):
        return memory.load_memory_variables({})["history"]

    chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough(),
            history=RunnableLambda(load_memory),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    paint_history()

    question = st.chat_input("문서에 대해 질문해 보세요.")
    if question:
        with st.chat_message("human"):
            st.markdown(question)
        save_message(question, "human")

        with st.chat_message("ai"):
            try:
                answer = st.write_stream(chain.stream(question))
            except Exception as e:
                st.error(f"답변을 만들지 못했어요.\n\n{e}")
                st.stop()
        save_message(answer, "ai")
        memory.save_context({"input": question}, {"output": answer})
