# -*- coding: utf-8 -*-
import streamlit as st
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from duckduckgo_search import DDGS

# ========================= 静态知识库（仅作备用）=========================
STATIC_KNOWLEDGE = [
    "凯文·杜兰特 1988年9月29日出生，绰号'KD'、'死神'。",
    "2007年NBA选秀榜眼，最佳新秀。",
    "2014年MVP，4次得分王。",
    "2017、2018年勇士队总冠军+FMVP。",
    "2019年跟腱断裂。",
    "2021年奥运金牌。",
    "生涯总得分历史前十。",
    "2023年被交易至太阳队。",  # 此条可能过时，但会被实时搜索覆盖
]

class TFIDFRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(documents)
    def get_relevant_documents(self, query, k=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_indices]

@st.cache_resource
def get_retriever():
    return TFIDFRetriever(STATIC_KNOWLEDGE)

# ========================= 实时搜索（优先）=========================
def search_duckduckgo(query):
    """搜索杜兰特最新信息，返回字符串摘要"""
    try:
        with DDGS() as ddgs:
            # 限定搜索关键词，获取新闻类结果
            results = list(ddgs.text(f"凯文杜兰特 最新消息 {query}", max_results=3))
            if not results:
                # 尝试英文搜索
                results = list(ddgs.text(f"Kevin Durant latest news {query}", max_results=3))
            if results:
                snippets = [f"- {r['body']}" for r in results]
                return "【实时搜索结果】\n" + "\n".join(snippets)
            else:
                return "【实时搜索结果】未找到最新相关信息。"
    except Exception as e:
        return f"【实时搜索结果】搜索出错: {str(e)}"

# ========================= 输出解释器 =========================
class KDStyleOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "抱歉，我现在无法思考清楚。请再问一次！🏀"
        # 移除可能重复的 "KD 说道：" 前缀
        if cleaned.startswith("KD 说道："):
            cleaned = cleaned[7:].strip()
        return f"🏀 **KD** 说道：\n\n{cleaned}\n\n---\n*#EasyMoneySniper* 🎯"

# ========================= 构建 Chain（实时搜索优先）=========================
def build_chain(retriever, memory, output_parser, llm, enable_search):
    prompt_template = """
你是一位精通篮球的AI助手，专门以篮球巨星凯文·杜兰特（Kevin Durant）的身份或视角回答问题。

【重要指令】
- 你将收到【实时搜索结果】和【本地知识库参考】两部分信息。
- 如果【实时搜索结果】中包含与【本地知识库参考】矛盾的最新信息（例如球队归属、最新交易等），你必须**无条件采用【实时搜索结果】的内容**。
- 你的回答应当以实时搜索结果为准，体现最新动态。如果实时搜索结果不足，再参考本地知识库或你自己的常识。

--- 以下为输入信息 ---

【实时搜索结果】（最新，优先级最高）
{realtime_context}

【本地知识库参考】（可能过时，仅作辅助）
{context}

对话历史：
{history}

用户最新问题：{question}

请用杜兰特的第一人称口吻自然、热情地回答，可以加一点KD标志性词语（如“easy money”）。
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    def get_history(_):
        return memory.load_memory_variables({})["history"]

    def retrieve_context(question):
        docs = retriever.get_relevant_documents(question)
        return "\n\n".join(docs)

    def get_realtime(question):
        if enable_search:
            return search_duckduckgo(question)
        else:
            return "（实时搜索未启用）"

    chain = (
        RunnablePassthrough.assign(
            history=get_history,
            context=lambda x: retrieve_context(x["question"]),
            realtime_context=lambda x: get_realtime(x["question"])
        )
        | prompt
        | llm
        | output_parser
    )
    return chain

# ========================= Streamlit 界面 =========================
def main():
    st.set_page_config(page_title="KD AI - 实时优先RAG", page_icon="🏀", layout="wide")

    with st.sidebar:
        st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/201142.png", width=100, caption="Kevin Durant")
        st.title("⚙️ 设置")
        api_key = st.text_input("DeepSeek API Key", type="password")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model_name = st.text_input("模型名称", value="deepseek-chat")
        base_url = st.text_input("API Base URL", value="https://api.deepseek.com/v1")
        temperature = st.slider("temperature", 0.0, 1.5, 0.7)
        enable_search = st.checkbox("🌐 启用实时搜索（获取最新消息）", value=True)
        if st.button("🧹 清空对话"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
        st.info("实时搜索优先于本地知识库，可回答最新交易、比赛等动态。")

    st.title("🏀 凯文·杜兰特 AI 助手（实时优先）")
    st.caption("开启实时搜索后，我会联网查找杜兰特的最新消息，并优先采用。")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="history")

    if not api_key:
        st.warning("请输入 DeepSeek API Key")
        st.stop()

    retriever = get_retriever()
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key, base_url=base_url)
    parser = KDStyleOutputParser()
    chain = build_chain(retriever, st.session_state.memory, parser, llm, enable_search)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏀" if msg["role"]=="assistant" else None):
            st.markdown(msg["content"])

    if prompt := st.chat_input("提问（例如：杜兰特现在在哪支球队？）"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🏀"):
            with st.spinner("正在联网搜索最新动态..." if enable_search else "思考中..."):
                try:
                    response = chain.invoke({"question": prompt})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(response)
                except Exception as e:
                    st.error(f"错误: {e}")

if __name__ == "__main__":
    main()
