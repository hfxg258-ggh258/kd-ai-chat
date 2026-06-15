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
import json

# ========== 静态知识库（不含球队信息）==========
STATIC_KNOWLEDGE = [
    "凯文·杜兰特 1988年9月29日出生，绰号'KD'、'死神'。",
    "2007年NBA选秀榜眼，最佳新秀。",
    "2014年MVP，4次得分王。",
    "2017、2018年勇士队总冠军+FMVP。",
    "2019年跟腱断裂。",
    "2021年奥运金牌。",
    "生涯总得分历史前十。",
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

# ========== 自动搜索最新球队信息 ==========
def search_current_team(api_key, base_url, model):
    """使用 DuckDuckGo 搜索 + 大模型提取杜兰特当前球队"""
    try:
        # 1. 搜索新闻
        with DDGS() as ddgs:
            results = list(ddgs.text("凯文杜兰特 现效力球队 2026", max_results=3))
            if not results:
                results = list(ddgs.text("Kevin Durant current team 2026", max_results=3))
            if not results:
                return "未知（搜索无结果）"
            news_snippets = "\n".join([r['body'] for r in results])
        
        # 2. 调用 DeepSeek 提取球队名称
        extraction_prompt = f"""
从以下新闻片段中提取凯文·杜兰特（Kevin Durant）目前效力的篮球队名称（例如：菲尼克斯太阳队、休斯顿火箭队、布鲁克林篮网队等）。
如果多处信息矛盾，以最新日期的为准；如果无法确定，回答“未知”。
只输出球队全称，不要输出任何解释。

新闻片段：
{news_snippets}
"""
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": extraction_prompt}],
            "temperature": 0.1
        }
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        if resp.status_code == 200:
            team = resp.json()["choices"][0]["message"]["content"].strip()
            if team and len(team) < 50:
                return team
        return "未知"
    except Exception as e:
        return f"获取失败: {str(e)}"

# ========== 输出解释器 ==========
class KDStyleOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "抱歉，我现在无法思考清楚。请再问一次！🏀"
        if cleaned.startswith("KD 说道："):
            cleaned = cleaned[7:].strip()
        return f"🏀 **KD** 说道：\n\n{cleaned}\n\n---\n*#EasyMoneySniper* 🎯"

# ========== 构建 Chain（自动球队信息注入）==========
def build_chain(retriever, memory, parser, llm, current_team):
    system_prompt = f"""
你以凯文·杜兰特的第一人称回答问题。
【最新事实】根据刚刚更新的信息，你当前效力的球队是：**{current_team}**。
如果用户问你现在在哪支球队，你必须回答 {current_team}。
除了球队信息外，其他关于你的生涯、荣誉等知识可以基于以下检索结果：
{{context}}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
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

# ========== 主界面 ==========
def main():
    st.set_page_config(page_title="KD AI - 自动实时更新", page_icon="🏀", layout="wide")
    with st.sidebar:
        st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/201142.png", width=100)
        st.title("⚙️ 设置")
        api_key = st.text_input("DeepSeek API Key", type="password")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model = st.text_input("模型", value="deepseek-chat")
        url = st.text_input("API地址", value="https://api.deepseek.com/v1")
        temp = st.slider("temperature", 0.0, 1.5, 0.3)
        
        # 自动更新球队信息按钮
        if st.button("🔄 自动获取最新球队信息", use_container_width=True):
            with st.spinner("正在搜索杜兰特最新动态..."):
                team = search_current_team(api_key, url, model)
                st.session_state.current_team = team
                st.success(f"已更新：{team}")
        
        if st.button("清空对话"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
        st.info("提示词模板 | 输出解释器 | Chain链 | Memory | RAG\n\n🤖 球队信息自动联网获取")
    st.title("🏀 凯文·杜兰特 AI 助手（实时自动更新）")
    # 初始化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="history")
    if "current_team" not in st.session_state:
        st.session_state.current_team = "未获取，请点击侧边栏按钮"
    if not api_key:
        st.warning("请输入 API Key")
        st.stop()
    retriever = get_retriever()
    llm = ChatOpenAI(model=model, temperature=temp, api_key=api_key, base_url=url)
    parser = KDStyleOutputParser()
    chain = build_chain(retriever, st.session_state.memory, parser, llm, st.session_state.current_team)
    # 显示当前球队状态
    st.caption(f"📌 当前知识库球队：**{st.session_state.current_team}** （点击侧边栏按钮自动更新）")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏀" if msg["role"]=="assistant" else None):
            st.markdown(msg["content"])
    if prompt := st.chat_input("例如：你现在在哪支球队？"):
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
