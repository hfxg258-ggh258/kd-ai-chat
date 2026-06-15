# -*- coding: utf-8 -*-
"""
凯文·杜兰特 AI 对话助手
技术栈：提示词模板、输出解释器、Chain链、Memory、RAG（本地TF-IDF）
基于 LangChain 0.3.0 + Streamlit
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

# ========================= 1. 本地 RAG 检索器（TF-IDF）=========================
KD_KNOWLEDGE = [
    "凯文·杜兰特 1988年9月29日出生于美国华盛顿特区，绰号'KD'、'死神'。",
    "2007年NBA选秀以榜眼身份被西雅图超音速队选中，获最佳新秀。",
    "2014年荣获NBA常规赛MVP，演讲'你才是真正的MVP'感人至深。",
    "4次获得NBA得分王，历史顶级得分手。",
    "2016年加盟金州勇士队，2017、2018年连续夺冠并获总决赛MVP。",
    "2019年总决赛跟腱断裂重伤，康复后仍保持巅峰状态。",
    "2021年助美国男篮夺东京奥运会金牌，成队史奥运得分王。",
    "生涯荣誉：2次总冠军、2次FMVP、1次MVP、4次得分王、13次全明星。",
    "身高208cm，臂展225cm，招牌干拔跳投和变向突破。",
    "名言：'Hard work beats talent when talent fails to work hard.'",
    "2023年被交易至菲尼克斯太阳队，与布克、比尔组成三巨头。",
    "生涯总得分历史前十，季后赛关键球能力顶级。",
    "帮助美国队夺得4枚奥运金牌。"
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

# 缓存检索器实例
@st.cache_resource
def get_retriever():
    return TFIDFRetriever(KD_KNOWLEDGE)

# ========================= 2. 输出解释器 =========================
class KDStyleOutputParser(StrOutputParser):
    """自定义输出解析器，添加 KD 风格前缀和标签"""
    def parse(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "抱歉，我现在无法思考清楚。请再问一次！🏀"
        return f"🏀 **KD** 说道：\n\n{cleaned}\n\n---\n*#EasyMoneySniper* 🎯"

# ========================= 3. 构建 Chain =========================
def build_chain(retriever, memory, output_parser, llm):
    # 提示词模板（包含系统消息、历史占位符、用户问题）
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
你是一位精通篮球的AI助手，专门以篮球巨星凯文·杜兰特（Kevin Durant）的身份或视角回答问题。
请严格基于以下【检索到的相关知识】进行回答。如果检索内容不足以回答问题，可以结合你自己的常识，但不要编造明显错误的事实。

【检索到的相关知识】
{context}

"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    # 从 memory 中提取历史消息
    def get_history(_):
        return memory.load_memory_variables({})["history"]

    # 从检索器获取上下文
    def retrieve_context(question):
        docs = retriever.get_relevant_documents(question)
        return "\n\n".join(docs)

    # LCEL 链式调用
    chain = (
        RunnablePassthrough.assign(
            history=get_history,
            context=lambda x: retrieve_context(x["question"])
        )
        | prompt
        | llm
        | output_parser
    )
    return chain

# ========================= 4. Streamlit 界面 =========================
def main():
    st.set_page_config(page_title="KD AI - RAG增强版", page_icon="🏀", layout="wide")

    # 侧边栏配置
    with st.sidebar:
        st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/201142.png", width=100, caption="Kevin Durant")
        st.title("⚙️ 模型设置")
        api_key = st.text_input("DeepSeek API Key", type="password",
                                placeholder="不填则读取环境变量 DEEPSEEK_API_KEY")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model_name = st.text_input("模型名称", value="deepseek-chat")
        base_url = st.text_input("API Base URL", value="https://api.deepseek.com/v1")
        temperature = st.slider("temperature", 0.0, 1.5, 0.7)
        if st.button("🧹 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
        st.divider()
        st.markdown("**🏆 技术栈**")
        st.info("✅ 提示词模板\n✅ 输出解释器\n✅ Chain链 (LCEL)\n✅ 对话记忆 (Memory)\n✅ RAG (TF-IDF 本地检索)")

    st.title("🏀 凯文·杜兰特 AI 助手（RAG增强版）")
    st.caption("知识库包含杜兰特生涯数据，使用本地 TF-IDF 检索增强生成。")

    # 初始化 session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="history")

    # 检查 API Key
    if not api_key:
        st.warning("⚠️ 请在上方侧边栏输入 DeepSeek API Key 以开始对话。")
        st.stop()

    # 初始化检索器
    retriever = get_retriever()

    # 初始化 LLM 和 Chain
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )
    parser = KDStyleOutputParser()
    chain = build_chain(retriever, st.session_state.memory, parser, llm)

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏀" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

    # 用户输入
    if prompt := st.chat_input("请输入关于凯文·杜兰特的问题，例如：杜兰特跟腱断裂是哪一年？"):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 Chain
        with st.chat_message("assistant", avatar="🏀"):
            with st.spinner("检索知识库并生成回答..."):
                try:
                    response = chain.invoke({"question": prompt})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    # 注意：memory 已经在 chain 内部自动更新？需要手动添加以确保同步
                    # 由于我们使用了 MessagesPlaceholder，memory 不会自动添加，需手动
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(response)
                except Exception as e:
                    error_msg = f"调用 API 失败: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(error_msg)

if __name__ == "__main__":
    main()
