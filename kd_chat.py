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

# ========== 1. 本地 RAG 检索器 ==========
KD_KNOWLEDGE = [
    "凯文·杜兰特 1988年9月29日出生，绰号'KD'、'死神'。",
    "2007年NBA选秀榜眼，最佳新秀。",
    "2014年MVP，4次得分王。",
    "2017、2018年勇士队总冠军+FMVP。",
    "2019年跟腱断裂。",
    "2021年奥运金牌。",
    "生涯总得分历史前十。",
    "2023年被交易至太阳队，后于2025年休赛期被交易至休斯顿火箭队。",  # 包含最新信息
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
    return TFIDFRetriever(KD_KNOWLEDGE)

# ========== 2. 输出解释器 ==========
class KDStyleOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "抱歉，我现在无法思考清楚。请再问一次！🏀"
        # 清理可能重复的前缀
        if cleaned.startswith("KD 说道："):
            cleaned = cleaned[7:].strip()
        return f"🏀 **KD** 说道：\n\n{cleaned}\n\n---\n*#EasyMoneySniper* 🎯"

# ========== 3. 构建 Chain ==========
def build_chain(retriever, memory, parser, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你以凯文·杜兰特的第一人称回答问题。基于以下检索到的知识：\n{context}\n"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    def get_history(_):
        return memory.load_memory_variables({})["history"]
    def retrieve(q):
        docs = retriever.get_relevant_documents(q)
        return "\n\n".join(docs)
    chain = (
        RunnablePassthrough.assign(
            history=get_history,
            context=lambda x: retrieve(x["question"])
        )
        | prompt
        | llm
        | parser
    )
    return chain

# ========== 4. 主界面 ==========
def main():
    st.set_page_config(page_title="KD AI - RAG增强版", page_icon="🏀", layout="wide")
    with st.sidebar:
        st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/201142.png", width=100)
        st.title("⚙️ 设置")
        api_key = st.text_input("DeepSeek API Key", type="password")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model = st.text_input("模型", value="deepseek-chat")
        url = st.text_input("API地址", value="https://api.deepseek.com/v1")
        temp = st.slider("temperature", 0.0, 1.5, 0.7)
        if st.button("清空对话"):
            st.session_state.clear()
            st.rerun()
        st.info("提示词模板 | 输出解释器 | Chain链 | Memory | RAG")
    st.title("🏀 凯文·杜兰特 AI 助手")
    st.caption("基于本地RAG（TF-IDF）+ 对话记忆")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="history")
    if not api_key:
        st.warning("请输入 API Key")
        st.stop()
    retriever = get_retriever()
    llm = ChatOpenAI(model=model, temperature=temp, api_key=api_key, base_url=url)
    parser = KDStyleOutputParser()
    chain = build_chain(retriever, st.session_state.memory, parser, llm)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏀" if msg["role"]=="assistant" else None):
            st.markdown(msg["content"])
    if prompt := st.chat_input("问吧"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🏀"):
            with st.spinner("思考中..."):
                try:
                    res = chain.invoke({"question": prompt})
                    st.markdown(res)
                    st.session_state.messages.append({"role": "assistant", "content": res})
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(res)
                except Exception as e:
                    st.error(f"错误: {e}")
if __name__ == "__main__":
    main()
