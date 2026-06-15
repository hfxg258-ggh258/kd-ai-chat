# -*- coding: utf-8 -*-
"""
凯文·杜兰特 AI 对话助手
技术栈：提示词模板、输出解释器、Chain链、Memory、RAG（本地TF-IDF）+ 可选实时搜索
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

# 尝试导入实时搜索库（如未安装，给出提示）
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

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

@st.cache_resource
def get_retriever():
    return TFIDFRetriever(KD_KNOWLEDGE)

# ========================= 2. 实时搜索功能 =========================
def search_realtime(query):
    """使用 DuckDuckGo 搜索关于凯文·杜兰特的最新信息"""
    if not DDGS_AVAILABLE:
        return "实时搜索库未安装，请运行 pip install duckduckgo-search"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"Kevin Durant {query} 凯文杜兰特 最新", max_results=3))
            if not results:
                return "未找到相关实时信息。"
            snippets = [r["body"] for r in results]
            return "\n\n".join(snippets)
    except Exception as e:
        return f"搜索出错: {str(e)}"

# ========================= 3. 输出解释器 =========================
class KDStyleOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "抱歉，我现在无法思考清楚。请再问一次！🏀"
        return f"🏀 **KD** 说道：\n\n{cleaned}\n\n---\n*#EasyMoneySniper* 🎯"

# ========================= 4. 构建 Chain =========================
def build_chain(retriever, memory, output_parser, llm, enable_search):
    # 提示词模板（系统消息 + 历史 + 检索上下文 + 实时搜索结果）
    system_template = """
你是一位精通篮球的AI助手，专门以篮球巨星凯文·杜兰特（Kevin Durant）的身份或视角回答问题。
请严格基于以下信息回答，优先级：实时搜索结果（最新） > 本地知识库 > 你自己的常识。

【本地知识库检索结果】
{context}

【实时搜索结果（来自网络，可能是最新动态）】
{realtime_context}

"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
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
            return search_realtime(question)
        else:
            return "（未启用实时搜索）"

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

# ========================= 5. Streamlit 界面 =========================
def main():
    st.set_page_config(page_title="KD AI - 实时RAG增强版", page_icon="🏀", layout="wide")

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
        
        # 实时搜索开关
        enable_search = st.checkbox("🌐 启用实时搜索（获取最新新闻/比赛数据）", value=True)
        if enable_search and not DDGS_AVAILABLE:
            st.error("⚠️ 实时搜索库未安装，请运行 `pip install duckduckgo-search`")
            enable_search = False
        
        if st.button("🧹 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
        st.divider()
        st.markdown("**🏆 技术栈**")
        st.info("✅ 提示词模板\n✅ 输出解释器\n✅ Chain链 (LCEL)\n✅ 对话记忆\n✅ 本地RAG (TF-IDF)\n✅ 实时搜索 (DuckDuckGo)")

    st.title("🏀 凯文·杜兰特 AI 助手（实时搜索增强）")
    st.caption("本地知识库 + 实时联网搜索，回答关于杜兰特的最新动态和历史数据。")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="history")

    if not api_key:
        st.warning("⚠️ 请在上方侧边栏输入 DeepSeek API Key 以开始对话。")
        st.stop()

    retriever = get_retriever()
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key, base_url=base_url)
    parser = KDStyleOutputParser()
    chain = build_chain(retriever, st.session_state.memory, parser, llm, enable_search)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏀" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

    if prompt := st.chat_input("请输入关于凯文·杜兰特的问题，例如：杜兰特今天有比赛吗？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🏀"):
            with st.spinner("检索知识库并搜索最新信息..."):
                try:
                    response = chain.invoke({"question": prompt})
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
