import ipaddress
import json
import os
import socket
from urllib.parse import urlparse

os.environ.setdefault("USER_AGENT", "research-assistant/1.0")

import streamlit as st
from openai import OpenAI
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader

GITHUB_URL = "https://github.com/nambak/fullstack-gpt"
MODEL = "gpt-5.6-terra"
MAX_TOOL_ROUNDS = 8

INSTRUCTIONS = """
당신은 꼼꼼한 리서치 어시스턴트입니다.
사용자가 어떤 주제의 조사를 요청하면 다음 순서로 진행하세요.
1. wikipedia_search와 duckduckgo_search로 주제를 검색합니다. 둘 중 하나가 실패하면 다른 하나로 계속하세요.
2. duckduckgo_search 결과에서 믿을 만한 웹사이드 1~2개를 골라 scrape_website로 내용을 읽습니다.
3. 모은 내용을 한국어로 정리합니다. 개요, 주요 사실(날짜, 버전, 관련 인물 등), 영향, 출처 URL 목록을 넣으세요.
4. save_research로 정리한 내용을 .txt 파일에 저장한 뒤, 정리한 내용을 답변으로도 보여 주세요.
조사 요청이 아닌 질문에는 지금까지의 대화를 바탕으로 한국어로 답하세요.
"""

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000)
)
duckduckgo = DuckDuckGoSearchResults(output_format="list", max_results=5)


def wikipedia_search(query):
    try:
        return wikipedia.invoke(query)
    except Exception as e:
        return f"위키피디아 검색에 실패했습니다: {e}"


def duckduckgo_search(query):
    try:
        results = duckduckgo.invoke(query)
    except Exception as e:
        return f"DuckDuckGo 검색에 실패했습니다: {e}"
    return "\n\n".join(
        f"제목: {r['title']}\n링크: {r['link']}\n요약: {r['snippet']}" for r in results
    )


def is_public_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
        return True


def scrape_website(url):
    if not is_public_url(url):
        return f"읽을 수 없는 주소입니다: {url}"
    try:
        soup = WebBaseLoader(url).scrape()
    except Exception as e:
        return f"웹사이트를 읽지 못했습니다: {e}"
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    body = soup.find("main") or soup.find("article") or soup.body or soup
    text = " ".join(body.get_text(" ").split())
    return text[:5000]


def save_research(filename, content):
    os.makedirs("research", exist_ok=True)
    name = os.path.basename(filename) or "research"
    if not name.endswith(".txt"):
        name += ".txt"
    with open(os.path.join("research", name), "w", encoding="utf-8") as f:
        f.write(content)
    st.session_state["files"][name] = content
    return f"{name} 파일로 저장했어요. ({len(content)}자)"


TOOL_FUNCTIONS = {
    "wikipedia_search": wikipedia_search,
    "duckduckgo_search": duckduckgo_search,
    "scrape_website": scrape_website,
    "save_research": save_research,
}


def function_tool(name, description, properties):
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        "strict": True,
    }


TOOLS = [
    function_tool(
        "wikipedia_search",
        "위키피디아에서 주제를 검색해 관련 문서의 요약을 돌려준다. 개념이나 사건의 배경을 알아볼 때 쓴다.",
        {"query": {"type": "string", "description": "검색어"}},
    ),
    function_tool(
        "duckduckgo_search",
        "DuckDuckGo에서 웹을 검색해 결과마다 제목, 링크(URL), 요약을 돌려준다. 자세히 읽을 웹사이트를 찾을 때 쓴다.",
        {"query": {"type": "string", "description": "검색어"}},
    ),
    function_tool(
        "scrape_website",
        "웹사이트 URL을 받아 페이지의 본문 텍스트를 추출한다. DuckDuckGo에서 찾은 링크의 내용을 읽을 때 쓴다.",
        {"url": {"type": "string", "description": "읽을 웹페이지 주소"}},
    ),
    function_tool(
        "save_research",
        "조사한 내용을 .txt 파일로 저장한다. 조사를 모두 마친 뒤 마지막에 쓴다.",
        {
            "filename": {"type": "string", "description": "저장할 파일 이름 (예: xz_backdoor.txt)"},
            "content": {"type": "string", "description": "저장할 조사 내용 전체"},
        },
    ),
]


def run_tool(call):
    function = TOOL_FUNCTIONS.get(call.name)
    if function is None:
        return f"알 수 없는 도구: {call.name}"
    try:
        return function(**json.loads(call.arguments))
    except Exception as e:
        return f"도구를 실행하지 못했어요: {e}"


def short(value, limit=120):
    text = str(value).replace("n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def reset_conversation():
    st.session_state["conversation_id"] = None
    st.session_state["files"] = {}


if "conversation_id" not in st.session_state:
    reset_conversation()

st.set_page_config(page_title="Research Assistant")
st.title("Research Assistant")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    if st.session_state.get("api_key") != api_key:
        st.session_state["api_key"] = api_key
        reset_conversation()
    st.button("새 대화 시작", on_click=reset_conversation, width="stretch")
    st.markdown("**예시 질문**")
    if st.button("Research about the XZ backdoor", width="stretch"):
        st.session_state["pending_question"] = "Research about the XZ backdoor"
    st.markdown(f"[GitHub 저장소 보기]({GITHUB_URL})")

if not api_key:
    st.info("왼쪽 사이드바에 OpenAI API 키를 입력해 주세요.")
    st.stop()

client = OpenAI(api_key=api_key)


def paint_history():
    conversation_id = st.session_state["conversation_id"]
    if not conversation_id:
        st.markdown("조사하고 싶은 주제를 입력해 보세요. 위키피디아와 DuckDuckGo에서 찾아 정리하고 파일로 저장합니다.")
        return
    for item in client.conversations.items.list(conversation_id, order="acs", limit=100):
        if item.type == "message" and item.role in ("user", "assistant"):
            text = "".join(getattr(part, "text", "") or "" for part in item.content)
            with st.chat_message("human" if item.role == "user" else "ai"):
                st.markdown(text)
        elif item.type == "function_call":
            with st.chat_message("ai"):
                st.caption(f"{item.name} 호출 -> {short(item.arguments)}")


def ask(question):
    if not st.session_state["conversation_id"]:
        conversation = client.conversations.create(metadata={"app": "research-assistant"})
        st.session_state["conversation_id"] = conversation.id
    conversation_id = st.session_state["conversation_id"]

    def create_response(input_items, tool_choice="auto"):
        return client.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,
            tools=TOOLS,
            tool_choice=tool_choice,
            conversation=conversation_id,
            input=input_items,
        )

    status = st.status("생각하는 중...", expanded=True)
    response = create_response([{"role": "user", "content": question}])
    rounds = 0
    while True:
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            break
        rounds += 1
        outputs = []
        for call in calls:
            status.write(f"**{call.name}** 실행 -> {short(call.arguments, 80)}")
            result = run_tool(call)
            status.write(f"{short(result)}")
            outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": result})
        response = create_response(outputs, tool_choice="none" if rounds >= MAX_TOOL_ROUNDS else "auto")
    status.update(label=f"완료 (도구 사용 {rounds}회)", state="complete", expanded=False)
    st.markdown(response.output_text or "답변이 비어 있습니다.")


try:
    paint_history()
except Exception as e:
    st.error(f"대화 기록을 불러오지 못했습니다. API 키를 확인하거나 '새 대화 시작'을 눌러 주세요\n\n{e}")
    st.stop()

question = st.chat_input("조사할 주제나 질문을 입력하세요.")
if not question:
    question = st.session_state.pop("pending_question", None)

if question:
    with st.chat_message("human"):
        st.markdown(question)
    with st.chat_message("ai"):
        try:
            ask(question)
        except Exception as e:
            st.error(f"답변을 만들지 못했습니다. 계속 오류가 나면 '새 대회 시작'을 눌러 주세요.\n\n{e}")

if st.session_state["files"]:
    with st.sidebar:
        st.markdown("**저장된 조사 파일**")
        for name, content in st.session_state["files"].items():
            st.download_button(f"{name}", content, file_name=name, mime="text/plain", width="stretch")
