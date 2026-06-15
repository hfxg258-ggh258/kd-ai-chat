# -*- coding: utf-8 -*-
"""
凯文·杜兰特 AI 对话助手
必备：提示词模板、输出解释器、Chain链
进阶：Memory对话记忆、RAG检索增强
"""

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

# ==================== 1. 本地 RAG 检索器（TF-IDF）====================
# 静态知识库（不包含动态球队信息，避免冲突）
STATIC_KNOWLEDGE = [
    "凯文·杜兰特 1988年9月29日出生，绰号'KD'、'死神'。",
    "2007年NBA选秀榜眼，最佳新秀。",
    "2014年常规赛MVP，4次得分王。",
    "2017、2018年勇士队总冠军并蝉联FMVP。",
    "2019年总决赛跟腱断裂，后完美康复。",
    "2021年东京奥运会金牌，美国队史奥运得分王。",
    "生涯总得分历史前十，季后赛关键球能力顶级。",
    "技术特点：身高208cm，臂展225cm，干拔跳投无解。",
    "名言：'Hard work beats talent when talent fails to work hard.'",
]


class TFIDFRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(docs)

    def get_relevant_documents(self, query, k=3):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_vectors).flatten()
        top_idx = np.argsort(scores)[-k:][::-1]
        return [self.docs[i] for i in top_idx]


@st.cache_resource
def get_retriever():
    return TFIDFRetriever(STATIC_KNOWLEDGE)


# ==================== 2. 输出解释器 ====================
class KDStyleOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "抱歉，我现在无法思考清楚。请再问一次！🏀"
        # 移除可能的重复前缀
        if cleaned.startswith("KD 说道："):
            cleaned = cleaned[7:].strip()
        return f"🏀 **KD** 说道：\n\n{cleaned}\n\n---\n*#EasyMoneySniper* 🎯"


# ==================== 3. 构建 Chain（含提示词模板、记忆、RAG）====================
def build_chain(retriever, memory, parser, llm):
    # 动态提示词模板（球队信息可从 session 或固定值获取）
    # 为了让用户可自行更新，我们在侧边栏提供球队输入框
    system_template = """
你以凯文·杜兰特的第一人称回答问题。
关于你的事实（除球队外）请基于以下检索结果：
{context}

【重要】如果你被问到“你现在在哪支球队”，请回答：**{current_team}**。
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    def get_history(_):
        return memory.load_memory_variables({})["history"]

    def retrieve(q):
        docs = retriever.get_relevant_documents(q)
        return "\n\n".join(docs)

    # LCEL 链
    chain = (
            RunnablePassthrough.assign(
                history=get_history,
                context=lambda x: retrieve(x["question"]),
                current_team=lambda x: x.get("current_team", "休斯顿火箭队")  # 默认值
            )
            | prompt
            | llm
            | parser
    )
    return chain


# ==================== 4. 主界面 ====================
def main():
    st.set_page_config(page_title="KD AI - 最强RAG", page_icon="🏀", layout="wide")

    # 侧边栏配置
    with st.sidebar:
        st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/201142.png", width=100)
        st.title("⚙️ 模型与知识库")

        # API 配置
        api_key = st.text_input("DeepSeek API Key", type="password",
                                placeholder="不填则读取环境变量 DEEPSEEK_API_KEY")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model_name = st.text_input("模型名称", value="deepseek-chat")
        base_url = st.text_input("API Base URL", value="https://api.deepseek.com/v1")
        temperature = st.slider("temperature", 0.0, 1.5, 0.3)

        st.divider()

        # 手动设置球队（保证答案正确，也可改为自动搜索）
        st.subheader("🏀 当前球队（可手动更新）")
        current_team = st.text_input("球队名称", value="休斯顿火箭队")
        if st.button("✅ 更新球队信息"):
            st.session_state.current_team = current_team
            st.success(f"已更新：{current_team}")

        st.divider()
        if st.button("🧹 清空对话"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()

        st.info("技术栈：提示词模板 | 输出解释器 | Chain链 | Memory | RAG(TF-IDF)")

    # 主区域
    st.title("🏀 凯文·杜兰特 AI 助手")
    st.caption("基于本地 RAG（TF-IDF）+ 对话记忆。球队信息可手动更新，确保回答准确。")

    # 初始化 session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="history")
    if "current_team" not in st.session_state:
        st.session_state.current_team = current_team  # 从输入框同步

    if not api_key:
        st.warning("⚠️ 请在上方侧边栏输入 DeepSeek API Key。")
        st.stop()

    # 显示当前球队状态
    st.info(f"📌 当前知识库中的球队：**{st.session_state.current_team}** （可在侧边栏修改）")

    # 构建组件
    retriever = get_retriever()
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key, base_url=base_url)
    parser = KDStyleOutputParser()
    chain = build_chain(retriever, st.session_state.memory, parser, llm)

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏀" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

    # 用户输入
    if prompt := st.chat_input("请问关于凯文·杜兰特的问题（如：你现在在哪支球队？）"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🏀"):
            with st.spinner("检索知识库并生成回答..."):
                try:
                    # 调用链，传入球队信息
                    response = chain.invoke({
                        "question": prompt,
                        "current_team": st.session_state.current_team
                    })
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(response)
                except Exception as e:
                    error_msg = f"调用失败: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(error_msg)


if __name__ == "__main__":
    main()