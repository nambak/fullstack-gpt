import os

from langchain_classic import embeddings
from langchain_community import vectorstores

os.environ.setdefault("USER_AGENT", "SiteGPT-Streamlit-App")

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import SitemapLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

GITHUB_URL = "https://github.com/nambak/fullstack-gpt"
SITEMAP_URL = "https://developers.cloudflare.com/sitemap-0.xml"
PRODUCT_FILTER = [r"^https://developers\.cloudflare\.com/(ai-geteway|vectorize|workers-ai)/.*"]
CHAT_MODEL = "gpt-5.6-terra"
EMBEDDING_MODEL = "text-embedding-3-small"
EXAMPLE_QUESTIONS = [
    "llama-2-7b-chat-fp16 모델의 1M 입력 토큰당 가격은 얼마인가요?",
    "Cloudflare의 AI 게이트웨이로 무엇을 할 수 있나요?",
    "벡터라이즈에서 단일 계정은 몇 개의 인덱스를 가질 수 있나요?",
]

st.set_page_config(page_title="SiteGPT")
st.title("SiteGPT for Cloudflare")

def parse_page(soup):
    main = soup.find("main")
    if main is None:
        return soup.get_text("\n", strip=True)
    for tag in main.find_all(["nav", "aside", "footer", "script", "style"]):
        tag.decompose()
    return main.get_text("\n", strip=True)

def add_metadata(meta, soup):
    title = soup.find("title")
    return {
        "source": meta["loc"],
        "title": title.get_text(strip=True) if title else "",
    }

@st.cache_resource(show_spinner="Cloudflare 문서를 불러오는 중...", ttl="1d")
def load_docs():
    loader = SitemapLoader(
        SITEMAP_URL,
        filter_urls=PRODUCT_FILTER,
        parsing_function=parse_page,
        meta_function=add_metadata,
    )
    loader.requests_per_second = 5
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.page_content = f"{chunk.metadata['title']}\n{chunk.page_content}"
    return chunks

@st.cache_resource(show_spinner="문서를 임베딩하는중...", max_entries=10)
def get_retriever(api_key):
    chunks = load_docs()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)
    cache_dir = LocalFileStore("./.cache/site_embeddings/")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
        namespace=EMBEDDING_MODEL,
        key_encoder="sha256"
    )
    vectorstore = FAISS.from_documents(chunks, cached_embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(f"[출처: {doc.metadata['source']}]\n{doc.page_content}" for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 Cloudflare 공식 문서(AI Gateway, Vectorize, Workers AI)에 대해 답하는 어시스턴트입니다.
    아래 문서 내용(context)만 사용해서 질문에 답하세요.
    문서에 없는 내용은 모른다고 답하고, 지어내지 마세요.
    가격이나 한도 같은 숫자는 문서에 적힌 그대로 단위와 함께 쓰시오.
    한국어로 답하세요
    
    {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)

def show_sources(sources):
    if sources:
        with st.expander("참고한 문서"):
            for url in sources:
                st.markdown(f"- {url}")

def paint_history():
    for item in st.session_state["messages"]:
        with st.chat_message(item["role"]):
            st.markdown(item["message"])
            show_sources(item.get("source"))

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("""
    ** 답변할 수 있는 문서**
    - [AI Gateway](https://developers.cloudflare.com/ai-gateway/)
    - [Vectorize](https://developers.cloudflare.com/vectorize/)
    - [Workers AI](https://developers.cloudflare.com/workers-ai/)
    """)
    st.markdown("**예시 질문**")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, width="stretch"):
            st.session_state["pending_question"] = example
    st.markdown(f"[GitHub 저장소 보기]({GITHUB_URL}")

if not api_key:
    st.info("왼쪽 사이드에 OpenAI API 키를 입력해 주세요.")
    st.stop()

try:
    retriever = get_retriever(api_key)
except Exception as e:
    st.error(f"문서를 준비하지 못했어요. API 키가 맞는지 확인해 주세요.\n\n{e}")
    st.stop()

paint_history()

question = st.chat_input("Cloudflare 문서에 대해 질문해 보세요")
if not question:
    question = st.session_state.pop("pending_question", None)

if question:
    with st.chat_message("human"):
        st.markdown(question)
    st.session_state["messages"].append({"message": question, "role": "human"})

    memory = st.session_state["memory"]
    llm = ChatOpenAI(model=CHAT_MODEL, openai_api_key=api_key)
    chain = prompt | llm | StrOutputParser()

    with st.chat_message("ai"):
        try:
            docs = retriever.invoke(question)
            answer = st.write_stream(chain.stream({
                "context": format_docs(docs),
                "question": question,
                "history": memory.load_memory_variables({})["history"],
            }))
        except Exception as e:
            st.error(f"답변을 만들지 못했어요.\n\n{e}")
            st.stop()
        sources = list(dict.fromkeys(doc.metadata["source"] for doc in docs))
        show_sources(sources)

    st.session_state["messages"].append({"message": answer, "role": "ai", "sources": sources})
    memory.save_context({"input": question}, {"output": answer})
