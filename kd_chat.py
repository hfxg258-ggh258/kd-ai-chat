# -*- coding: utf-8 -*-
"""
凯文·杜兰特 AI 对话助手
要求：提示词模板、输出解释器、Chain链、Memory、RAG
LangChain 0.3.0 + Streamlit
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
from langchain.memory import ConversationBufferMemory  # ✅ 在 langchain 0.3.0 中可用

# ========================= 1. 本地 RAG 检索器（TF-IDF）=========================
# 静态知识库（不含动态球队，避免过时）
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

# ========================= 2. 输出解释器 =========================
class KDStyleOutputParser(StrOutputParser):
    """自定义输出解析器，添加KD风格前缀和标签"""
    def parse(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "抱歉，我现在无法思考清楚。请再问一次！🏀"
        # 移除可能重复的标记
        if cleaned.startswith("KD 说道："):
            cleaned = cleaned[7:].strip()
        return f"🏀 **KD** 说道：\n\n{cleaned}\n\n---\n*#EasyMoneySniper* 🎯"

# ========================= 3. 构建 Chain =========================
def build_chain(retriever, memory, output_parser, llm, current_team):
    # 提示词模板：包含系统消息、历史占位符、检索上下文、动态球队
    system_template = """
你以凯文·杜兰特的第一人称回答问题。
请基于以下从知识库中检索到的相关信息（关于你的生涯、荣誉等）：
{context}

【重要】如果你被问到“你现在在哪支球队”或类似问题，请回答：**{current_team}**（这是经过确认的最新事实）。
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    # 从memory加载历史
    def get_history(_):
        return memory.load_memory_variables({})["history"]
    
    # 从检索器获取上下文
    def retrieve_context(question):
        docs = retriever.get_relevant_documents(question)
        return "\n\n".join(docs)
    
    # LCEL 链
    chain = (
        RunnablePassthrough.assign(
            history=get_history,
            context=lambda x: retrieve_context(x["question"]),
            current_team=lambda x: x.get("current_team", current_team)
        )
        | prompt
        | llm
        | output_parser
    )
    return chain

# ========================= 4. Streamlit 界面 =========================
def main():
    st.set_page_config(page_title="KD AI - 最终作业版", page_icon="🏀", layout="wide")
    
    with st.sidebar:
        st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/201142.png", width=100)
        st.title("⚙️ 模型与知识库")
        
        # DeepSeek API 配置
        api_key = st.text_input("DeepSeek API Key", type="password",
                                placeholder="不填则读取环境变量 DEEPSEEK_API_KEY")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model_name = st.text_input("模型名称", value="deepseek-chat")
        base_url = st.text_input("API Base URL", value="https://api.deepseek.com/v1")
        temperature = st.slider("temperature", 0.0, 1.5, 0.3)
        
        st.divider()
        
        # 手动更新球队信息（确保回答正确，避免自动搜索失败）
        st.subheader("🏀 当前球队（手动更新以确保准确）")
        current_team = st.text_input("请输入杜兰特现效力的球队", value="休斯顿火箭队")
        if st.button("✅ 确认更新"):
            st.session_state.current_team = current_team
            st.success(f"已更新：{current_team}")
        
        st.divider()
        
        if st.button("🧹 清空对话"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
        
        st.markdown("**技术栈实现**")
        st.info("✅ 提示词模板 (ChatPromptTemplate)\n✅ 输出解释器 (自定义Parser)\n✅ Chain链 (LCEL)\n✅ Memory (ConversationBufferMemory)\n✅ RAG (本地TF-IDF检索)")
    
    # 主界面
    st.title("🏀 凯文·杜兰特 AI 助手（完整技术栈）")
    st.caption("具备：提示词模板 | 输出解释器 | Chain链 | 对话记忆 | RAG增强检索")
    
    # 初始化 session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="history")
    if "current_team" not in st.session_state:
        st.session_state.current_team = current_team
    
    if not api_key:
        st.warning("⚠️ 请在上方侧边栏输入 DeepSeek API Key。")
        st.stop()
    
    # 显示当前球队状态
    st.info(f"📌 当前球队：**{st.session_state.current_team}** （如需更改，请在左侧修改）")
    
    # 构建组件
    retriever = get_retriever()
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key, base_url=base_url)
    parser = KDStyleOutputParser()
    chain = build_chain(retriever, st.session_state.memory, parser, llm, st.session_state.current_team)
    
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏀" if msg["role"]=="assistant" else None):
            st.markdown(msg["content"])
    
    # 用户输入
    if prompt := st.chat_input("请问关于凯文·杜兰特的问题（例如：你现在在哪支球队？）"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🏀"):
            with st.spinner("检索知识库并生成回答..."):
                try:
                    # 调用链
                    response = chain.invoke({"question": prompt})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    # 手动更新 memory（因为 chain 中的 MessagesPlaceholder 不会自动保存，需手动）
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
