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

# ==================== 1. 本地 RAG 检索器（TF-IDF）====================
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
        if cleaned.startswith("KD 说道："):
            cleaned = cleaned[7:].strip()
        return f"🏀 **KD** 说道：\n\n{cleaned}\n\n---\n*#EasyMoneySniper* 🎯"

# ==================== 3. 手动记忆管理 ====================
def format_memory(history_list):
    """将对话历史格式化为字符串，供提示词使用"""
    if not history_list:
        return ""
    formatted = ""
    for msg in history_list:
        role = "用户" if msg["role"] == "user" else "KD"
        formatted += f"{role}: {msg['content']}\n"
    return formatted

# ==================== 4. 构建 Chain（不使用 langchain.memory）====================
def build_chain(retriever, parser, llm):
    # 提示词模板：包含系统提示、对话历史占位符、检索上下文
    prompt_template = """
你以凯文·杜兰特的第一人称回答问题。
请严格基于以下检索到的相关知识（关于除球队外的其他事实）：
{context}

以下是最近的对话历史：
{history}

当前用户问：{question}

【重要】如果你被问到“你现在在哪支球队”，请回答：**{current_team}**。
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def retrieve(q):
        docs = retriever.get_relevant_documents(q)
        return "\n\n".join(docs)
    
    # LCEL 链
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: retrieve(x["question"]),
            history=lambda x: x["history"],
            current_team=lambda x: x.get("current_team", "休斯顿火箭队")
        )
        | prompt
        | llm
        | parser
    )
    return chain

# ==================== 5. 主界面 ====================
def main():
    st.set_page_config(page_title="KD AI - 最终版", page_icon="🏀", layout="wide")
    
    # 侧边栏配置
    with st.sidebar:
        st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/201142.png", width=100)
        st.title("⚙️ 设置")
        api_key = st.text_input("DeepSeek API Key", type="password",
                                placeholder="不填则读取环境变量 DEEPSEEK_API_KEY")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model_name = st.text_input("模型名称", value="deepseek-chat")
        base_url = st.text_input("API Base URL", value="https://api.deepseek.com/v1")
        temperature = st.slider("temperature", 0.0, 1.5, 0.3)
        
        st.divider()
        st.subheader("🏀 当前球队（手动更新）")
        current_team = st.text_input("球队名称", value="休斯顿火箭队")
        if st.button("✅ 更新球队信息"):
            st.session_state.current_team = current_team
            st.success(f"已更新：{current_team}")
        
        st.divider()
        if st.button("🧹 清空对话"):
            st.session_state.messages = []
            st.session_state.history = []  # 手动历史列表
            st.rerun()
        
        st.info("✅ 提示词模板\n✅ 输出解释器\n✅ Chain链(LCEL)\n✅ 手动记忆\n✅ RAG(TF-IDF)")
    
    # 主区域
    st.title("🏀 凯文·杜兰特 AI 助手（最终稳定版）")
    st.caption("本地 RAG + 手动记忆管理，完全规避版本冲突。")
    
    # 初始化 session
    if "messages" not in st.session_state:
        st.session_state.messages = []      # 用于显示
        st.session_state.history = []       # 用于记忆（存储 {"role": "user"/"assistant", "content": ...}）
    if "current_team" not in st.session_state:
        st.session_state.current_team = current_team
    
    if not api_key:
        st.warning("⚠️ 请输入 DeepSeek API Key。")
        st.stop()
    
    # 显示当前球队
    st.info(f"📌 当前球队设定：**{st.session_state.current_team}** （可在侧边栏修改）")
    
    # 构建组件
    retriever = get_retriever()
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key, base_url=base_url)
    parser = KDStyleOutputParser()
    chain = build_chain(retriever, parser, llm)
    
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏀" if msg["role"]=="assistant" else None):
            st.markdown(msg["content"])
    
    # 用户输入
    if prompt := st.chat_input("请问关于凯文·杜兰特的问题（如：你现在在哪支球队？）"):
        # 记录用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 调用 Chain
        with st.chat_message("assistant", avatar="🏀"):
            with st.spinner("检索知识库并生成回答..."):
                try:
                    # 格式化历史（取最近8条）
                    history_str = format_memory(st.session_state.history[-8:])
                    response = chain.invoke({
                        "question": prompt,
                        "history": history_str,
                        "current_team": st.session_state.current_team
                    })
                    st.markdown(response)
                    # 记录助手消息
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"调用失败: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.history.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
