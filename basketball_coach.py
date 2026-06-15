# -*- coding: utf-8 -*-
"""
篮球技术分析助手 - 使用LangChain + Streamlit
功能：提示词模板、输出解释器、Chain链、对话记忆(Memory)、RAG检索增强生成
主题：帮助分析球员投篮、运球、防守等技术动作
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

# ========== 1. 本地知识库（RAG用）==========
BASKETBALL_KNOWLEDGE = [
    "投篮技术要点：保持平衡，肘部内收，跟随动作完整，手腕下压。",
    "运球技术：低重心，用手指控球，抬头观察，变化节奏。",
    "防守技术：滑步保持重心，手部干扰但不犯规，预判传球路线。",
    "上篮技巧：保护球远离防守者，使用篮板，调整步幅。",
    "三分球：腿部发力充分，出手点高，弧线适中。",
    "中距离跳投：急停稳定，身体垂直，瞄准篮筐后沿。",
    "传球技术：胸前传球、击地传球、过顶传球，根据防守选择。",
    "篮板球：卡位、判断落点、双手高举、快速传出一传。",
    "脚步训练：交叉步、侧滑步、后退步，提高敏捷性。",
    "体能训练：折返跑、跳绳、核心力量，提升耐力和爆发力。",
    "实战战术：挡拆配合、无球掩护、快攻转换。",
    "心理素质：保持专注，不怕失误，关键时刻敢于出手。",
]

class TfidfRetriever:
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
    return TfidfRetriever(BASKETBALL_KNOWLEDGE)

# ========== 2. 输出解释器（添加篮球教练风格）==========
class CoachOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "抱歉，我没听清你的问题。再问一次吧！🏀"
        # 避免重复前缀
        if cleaned.startswith("教练说："):
            cleaned = cleaned[4:].strip()
        return f"🏀 **教练** 说道：\n\n{cleaned}\n\n---\n*#KeepTraining* 🧠"

# ========== 3. 构建Chain（含提示词模板、Memory、RAG）==========
def build_chain(retriever, memory, parser, llm):
    # 系统提示词模板
    system_template = """
你是一名经验丰富的篮球技术教练。你的任务是回答用户关于篮球技术、训练方法、战术等问题。
如果用户询问具体技术动作（如投篮、运球、防守等），请严格基于以下【相关知识库】回答：
{context}

对话历史：
{history}

【重要风格】：
- 用鼓励、专业的口吻，可以加入“保持专注”、“再来一组”、“注意细节”等教练常用语。
- 回答简洁实用，每一点建议都要可操作。
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

    # LCEL 链式调用
    chain = (
        RunnablePassthrough.assign(
            history=get_history,
            context=lambda x: retrieve_context(x["question"])
        )
        | prompt
        | llm
        | parser
    )
    return chain

# ========== 4. Streamlit界面 ==========
def main():
    st.set_page_config(page_title="篮球技术教练 - AI助手", page_icon="🏀", layout="wide")

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Basketball.png/120px-Basketball.png", width=80)
        st.title("⚙️ 模型设置")
        api_key = st.text_input("DeepSeek API Key", type="password",
                                placeholder="不填则读取环境变量 DEEPSEEK_API_KEY")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model_name = st.text_input("模型名称", value="deepseek-chat")
        base_url = st.text_input("API Base URL", value="https://api.deepseek.com/v1")
        temperature = st.slider("temperature", 0.0, 1.5, 0.5)
        if st.button("🧹 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
        st.divider()
        st.markdown("**🏆 技术栈**")
        st.info("✅ 提示词模板\n✅ 输出解释器\n✅ Chain链 (LCEL)\n✅ 对话记忆\n✅ RAG (TF-IDF检索)")

    st.title("🏀 篮球技术教练 AI")
    st.caption("问我任何关于篮球技术、训练、战术的问题，我会给你专业建议！")

    # 初始化 session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="history")

    if not api_key:
        st.warning("⚠️ 请在左侧输入 DeepSeek API Key 开始对话。")
        st.stop()

    # 初始化检索器和链
    retriever = get_retriever()
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key, base_url=base_url)
    parser = CoachOutputParser()
    chain = build_chain(retriever, st.session_state.memory, parser, llm)

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏀" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

    # 用户输入
    if prompt := st.chat_input("例如：如何提高投篮命中率？或者：运球时总低头怎么办？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🏀"):
            with st.spinner("正在检索训练方法..."):
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