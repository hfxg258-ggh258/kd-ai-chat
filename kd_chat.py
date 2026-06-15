# -*- coding: utf-8 -*-
import streamlit as st
import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun

# ===== 1. 自定义输出解释器 =====
class KDStyleOutputParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "抱歉，我现在无法思考清楚。请再问一次！🏀"
        return f"🏀 **KD** 说道：\n\n{cleaned}\n\n---\n*#EasyMoneySniper* 🎯"

# ===== 2. 构建 RAG 向量库（静态知识）=====
@st.cache_resource
def build_kd_vectorstore():
    kd_docs = [
        "凯文·杜兰特 (Kevin Durant) 1988年9月29日出生于美国华盛顿特区，司职小前锋/大前锋，绰号'KD'、'死神'。",
        "杜兰特在2007年NBA选秀中以榜眼身份被西雅图超音速队选中，新秀赛季获得最佳新秀。",
        "2014年，杜兰特荣获NBA常规赛最有价值球员(MVP)，他的MVP演讲'你才是真正的MVP'感动无数人。",
        "杜兰特职业生涯4次获得NBA得分王，被誉为历史顶级得分手，拥有无解的中距离和三分能力。",
        "2016年杜兰特加盟金州勇士队，连续两年（2017、2018）夺得NBA总冠军并获得总决赛MVP。",
        "2019年总决赛期间杜兰特遭遇跟腱断裂重伤，但康复后依然保持巅峰状态，展现了坚强的意志。",
        "2021年，杜兰特代表美国男篮夺得东京奥运会金牌，成为美国队史奥运得分王。",
        "杜兰特生涯荣誉：2次总冠军、2次FMVP、1次MVP、4次得分王、13次全明星、10次最佳阵容。",
        "杜兰特身高208cm，臂展225cm，控球技术出众，招牌动作是干拔跳投和变向突破。",
        "场外杜兰特热衷于慈善，曾捐赠数百万美元用于社区建设和教育项目。",
        "杜兰特经典名言：'Hard work beats talent when talent fails to work hard.' (天赋不努力，会被努力的天赋打败)",
        "KD 是社交媒体活跃者，经常和球迷互动，真实有趣，被球迷称为'小帅'。",
        "2023年杜兰特被交易至菲尼克斯太阳队，继续追逐总冠军，与布克、比尔组成三巨头。",
        "杜兰特职业生涯总得分历史前十，季后赛总得分也名列前茅，关键球能力顶级。",
        "2010年土耳其世锦赛、2012伦敦奥运、2016里约奥运、2020东京奥运，KD帮助美国队拿下4枚金牌。",
        "杜兰特在雷霆时期曾连续4年打入西部决赛，2012年率队打进总决赛。",
        "杜兰特的技术特点：Catch & Shoot 高效，突破节奏变化丰富，防守端利用身高臂展干扰投篮。"
    ]
    documents = [Document(page_content=doc) for doc in kd_docs]
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# ===== 3. 实时搜索函数（使用 DuckDuckGo，免费）=====
def search_latest_news(query: str) -> str:
    """
    搜索关于凯文·杜兰特的最新信息
    """
    try:
        search_tool = DuckDuckGoSearchRun()
        # 限定搜索关键词为杜兰特相关
        search_query = f"Kevin Durant 凯文杜兰特 最新 {query}"
        result = search_tool.invoke(search_query)
        # 限制结果长度，避免超出 token 限制
        if len(result) > 2000:
            result = result[:2000] + "...(已截断)"
        return result
    except Exception as e:
        return f"搜索时出错：{str(e)}。请稍后重试。"

# ===== 4. 判断是否需要实时搜索 =====
def need_realtime_search(question: str) -> bool:
    """
    根据问题关键词判断是否需要联网搜索最新信息
    """
    keywords = ["最新", "今天", "现在", "实时", "近期", "最近", "刚刚", "本赛季", "今日", "昨晚", "上一场", "比赛结果", "数据", "新闻", "消息", "更新"]
    return any(kw in question for kw in keywords)

# ===== 5. 初始化 LLM =====
def get_llm(api_key: str, model_name: str, temperature: float, base_url: str = None):
    llm_kwargs = {"model": model_name, "temperature": temperature, "api_key": api_key}
    if base_url:
        llm_kwargs["base_url"] = base_url
    return ChatOpenAI(**llm_kwargs)

# ===== 6. 构建完整 Chain（支持实时搜索）=====
def build_chain(retriever, memory, output_parser, llm):
    system_template = """
你是一位精通篮球的AI助手，专门以篮球巨星凯文·杜兰特（Kevin Durant）的身份或视角回答问题。
请严格基于以下【背景知识】和【实时搜索结果】进行回答。

- 如果【实时搜索结果】不为空，请优先使用其中的最新信息回答关于杜兰特近期动态、比赛数据或新闻的问题。
- 如果问题与最新实时信息无关，可以忽略搜索结果，主要依靠背景知识。
- 不要编造事实，如果信息不足，请诚实地说“我不太确定，建议查看最新的体育新闻”。

【背景知识】（来自历史资料）
{context}

【实时搜索结果】（来自网络，可能包含最新信息）
{realtime_context}

【对话历史】
{chat_history}

用户问题：{question}

请用自然、热情的口吻回答，可以加入一点KD风格的标志性词语（比如“easy money”、“你能防住我？”等调侃感），但必须准确传达信息。
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{question}")
    ])

    def get_chat_history(_inputs):
        if memory and hasattr(memory, "load_memory_variables"):
            mem_vars = memory.load_memory_variables({})
            if "history" in mem_vars:
                return get_buffer_string(mem_vars["history"])
        return ""

    def retrieve_context(question):
        if retriever:
            docs = retriever.invoke(question)
            return "\n\n".join([doc.page_content for doc in docs])
        return "暂无检索到的背景知识，请用你自身的篮球知识回答。"

    def get_realtime_context(question):
        if need_realtime_search(question):
            with st.spinner("🔍 正在搜索凯文·杜兰特的最新动态..."):
                return search_latest_news(question)
        return "（本次问题不涉及实时信息，未进行搜索）"

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: retrieve_context(x["question"]),
            realtime_context=lambda x: get_realtime_context(x["question"]),
            chat_history=lambda x: get_chat_history(x)
        )
        | prompt
        | llm
        | output_parser
    )
    return chain

# ===== 7. Streamlit 主界面 =====
def main():
    st.set_page_config(page_title="KD AI - 凯文·杜兰特专属助手（实时版）", page_icon="🏀", layout="wide")

    with st.sidebar:
        st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/201142.png", width=100, caption="Kevin Durant")
        st.title("⚙️ 模型设置")
        api_key = st.text_input("DeepSeek API Key / OpenAI Key", type="password",
                                placeholder="不填则读取环境变量 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        model_name = st.text_input("模型名称", value="deepseek-chat")
        base_url = st.text_input("API Base URL (可选)", placeholder="https://api.deepseek.com/v1",
                                 value="https://api.deepseek.com/v1")
        temperature = st.slider("temperature", 0.0, 1.5, 0.7)
        if st.button("🧹 清空当前对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
            st.rerun()
        st.divider()
        st.markdown("**🏆 关于KD AI（实时增强版）**")
        st.info(
            "本助手基于 LangChain + RAG + 实时搜索，能回答关于凯文·杜兰特的历史知识和最新动态。\n"
            "当您的问题包含「最新、今天、现在、实时」等关键词时，会自动联网搜索最新信息。"
        )
        st.markdown("Made with ❤️ by Basketball Fan")

    st.title("🏀 凯文·杜兰特 AI 对话助手（实时版）")
    st.caption("聊聊篮球，致敬KD — 你可以问历史知识，也可以问今天的最新比赛或新闻！")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    if "retriever" not in st.session_state:
        with st.spinner("🔥 加载KD知识库中 (首次运行会自动下载轻量模型)..."):
            st.session_state.retriever = build_kd_vectorstore()

    if not api_key:
        st.warning("⚠️ 请在上方侧边栏输入 DeepSeek API Key 或 OpenAI Key 以开始对话。")
        st.stop()

    llm = get_llm(api_key, model_name, temperature, base_url if base_url else None)
    output_parser = KDStyleOutputParser()
    chain = build_chain(st.session_state.retriever, st.session_state.memory, output_parser, llm)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🏀"):
                st.markdown(msg["content"])

    if prompt := st.chat_input("请输入关于凯文·杜兰特的问题（例如：杜兰特今天比赛数据？KD 最近有什么新闻？）"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🏀"):
            with st.spinner("KD 正在组织语言，easy money..."):
                try:
                    response = chain.invoke({"question": prompt})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(response)
                except Exception as e:
                    error_msg = f"抱歉，调用API时出错: {str(e)}。请检查API Key或网络设置。"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(error_msg)

if __name__ == "__main__":
    main()
