import hashlib
import json
import random

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate

GITHUB_URL = "https://github.com/nambak/fullstack-gpt"
CHAT_MODEL = "gpt-5.6-terra"
QUESTION_COUNT = 5
MAX_CONTEXT_CHARS = 20000

DIFFICULTY_GUIDE = {
    "쉬움": "문서에 그대로 나온 사실(인물, 장소, 숫자 등)을 묻는 쉬운 문제를 만드세요. 오답 보기는 정답과 확실히 구별되게 하세요.",
    "어려움": "여러 문장의 내용을 연결하거나 원인과 결과를 추론해야 풀 수 있는 어려운 문제를 만드세요. 오답 보기도 그럴듯해서 헷갈리게 하세요."
}

QUIZ_FUNCTION = {
    "name": "create_quiz",
    "description": "객관식 문제와 보기 목록을 받아 퀴즈를 만든다.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"},
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 학생들의 이해도를 확인하는 선생님입니다.
    아래 자료(context)만 사용해서 객관식 문제 {count}개를 만드세요.
    각 문제의 보기는 4개이고, 그중 정답은 정확히 1개입니다.
    난이도: {difficulty} = {guide}
    문제와 보기는 모둔 한국어로 쓰세요.
    만든 문제는 반드시 create_quiz 함수로 전달하세요.
    
    자료:
    {context}"""),
    ("human", "퀴즈를 만들어 주세요."),
])

def read_text_file(file):
    raw = file.getvalue()
    for encoding in ("utf-8", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")

@st.cache_data(show_spinner="위키피디아에서 찾는 중...")
def search_wikipedia(topic):
    retriever = WikipediaRetriever(top_k_results=3, lang="ko")
    docs = retriever.invoke(topic)
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_data(show_spinner="퀴즈를 만드는 중...")
def generate_quiz(context, difficulty, api_key):
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=api_key,
        use_responses_api=True,
    ).bind_tools(
        [QUIZ_FUNCTION],
        tool_choice="create_quiz",
    )
    chain = prompt | llm
    response = chain.invoke({
        "context": context[:MAX_CONTEXT_CHARS],
        "difficulty": difficulty,
        "guide": DIFFICULTY_GUIDE[difficulty],
        "count": QUESTION_COUNT,
    })
    if not response.tool_calls:
        raise ValueError("모델이 create_quiz 함수를 호출하지 않았어요.")

    questions = []
    for question in response.tool_calls[0]["args"]["questions"]:
        answers = question["answers"]
        if sum(answer["correct"] for answer in answers) != 1:
            continue
        random.shuffle(answers)
        questions.append(question)
    if not questions:
        raise ValueError("올바른 형식의 문제를 받지 못했어요.")
    return {"questions": questions}

def grade():
    quiz = st.session_state["quiz"]
    attempt = st.session_state["attempt"]
    picked = [st.session_state.get(f"q{i}_{attempt}") for i in range(len(quiz["questions"]))]
    if None in picked:
        st.session_state["result"] = {"incomplete": True}
        return
    marks = [
        any(answer["answer"] == choice and answer["correct"] for answer in question["answers"])
        for question, choice in zip(quiz["questions"], picked)
    ]
    st.session_state["result"] = {"marks": marks, "celebrated": False}

def retake():
    st.session_state["attempt"] += 1
    st.session_state["result"] = None

st.set_page_config(page_title="QuizGPT Turbo")
st.title("QuizGPT Turbo")

with st.sidebar:
    api_key = st.text_input("OpenAI API key", type="password")
    difficulty = st.selectbox("난이도", list(DIFFICULTY_GUIDE.keys()))
    source = st.radio("퀴즈 자료", ["파일", "위키피디아"], horizontal=True)

    context = None
    if source == "파일":
        file = st.file_uploader("텍스트 파일을 올려 주세요 (.txt)", type=["txt"])
        if file:
            context = read_text_file(file)
    else:
        topic = st.text_input("위키피디아에서 찾을 주제")
        if topic:
            try:
                context = search_wikipedia(topic)
            except Exception as e:
                st.error(f"위키피디아 검색에 실패했어요.\n\n{e}")
            if topic and not context:
                st.warning("검색 결과가 없어요. 다른 주제로 찾아보세요.")

    st.markdown(f"[🔗Github 저장소 보기]({GITHUB_URL})")

if not api_key:
    st.info("왼쪽 사이드바에 OpenAI API 키를 입력해 주세요.")
elif not context:
    st.markdown("""
    QuizGPT Turbo에 오신 걸 환영합니다!
    
    왼쪽 사이드바에서 **난이도**를 고르고, **파일을 올리거나 위키피디아 주제를 입력**하면 그 내용으로 퀴즈를 만들어 드려요.
    """)
else:
    try:
        new_quiz = generate_quiz(context, difficulty, api_key)
    except Exception as e:
        st.error(f"퀴즈를 만들지 못했어요. API 키가 맞는지 확인해 주세요.\n\n{e}")
        st.stop()

    quiz_id = hashlib.md5(json.dumps(new_quiz, ensure_ascii=False).encode()).hexdigest()
    if st.session_state.get("quiz_id") != quiz_id:
        st.session_state["quiz_id"] = quiz_id
        st.session_state["quiz"] = new_quiz
        st.session_state["attempt"] = 0
        st.session_state["result"] = None

    quiz = st.session_state["quiz"]
    attempt = st.session_state["attempt"]
    result = st.session_state["result"]
    graded = bool(result and "marks" in result)

    caption = f"난이도 {difficulty}·{len(quiz['questions'])}문제"
    if attempt:
        caption += f"·{attempt + 1}번쨰 도전"
    st.caption(caption)

    with st.form(f"quiz_form_{attempt}"):
        for i, question in enumerate(quiz["questions"]):
            st.markdown(f"**{i + 1}. {question['question']}**")
            st.radio(
                f"{i + 1}번 문제",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"q{i}_{attempt}",
                label_visibility="collapsed",
            )
            if graded:
                if result["marks"][i]:
                    st.success("정답입니다")
                else:
                    st.error("틀렸습니다")
        st.form_submit_button("제출하기", on_click=grade, disabled=graded)

    if result and result.get("incomplete"):
        st.warning("모든 문제에 답한 뒤 제출해 주세요.")
        st.session_state["result"] = None
    elif graded:
        score = sum(result["marks"])
        total = len(result["marks"])
        if score == total:
            if not result["celebrated"]:
                st.balloons()
                result["celebrated"] = True
            st.success(f"만점입니다. {total}문제를 모두 맞혔습니다.")
        else:
            st.warning(f"{total}문제 중 {score}문제를 맞혔어요. 다시 도전해 보세요.")
            st.button("다시 풀기", on_click=retake)

