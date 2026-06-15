# -*- coding: utf-8 -*-
import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# 关键修正：使用 langchain_classic 替代已废弃的 langchain.memory
from langchain_classic.memory import ConversationBufferMemory

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

# ===== 1. 自定义输出解释器 =====
class KDStyleOutputParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "抱歉，我现在无法思考清楚。请再问一次！🏀"
        return f"🏀 **KD** 说道：\n\n{cleaned}\n\n---\n*#EasyMoneySniper* 🎯"

# ===== 2. 构建 RAG 向量库 =====
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

# ===== 3. 初始化 LLM =====
def get_llm(api_key: str, model_name: str, temperature: float, base_url: str = None):
    llm_kwargs = {"model": model_name, "temperature": temperature, "api_key": api_key}
    if base_url:
        llm_kwargs["base_url"] = base_url
    return ChatOpenAI(**llm_kwargs)

# ===== 4. 构建完整 Chain =====
def build_chain(retriever, memory, output_parser, llm):
    system_template = """
你是一位精通篮球的AI助手，专门以篮球巨星凯文·杜兰特（Kevin Durant）的身份或视角回答问题。
请严格基于以下【背景知识】进行回答，如果背景知识不足，可以结合你自己的篮球知识，但不要编造关于杜兰特的关键事实。

【背景知识】
{context}

【对话历史】
{chat_history}

用户问题：{question}

请用自然、热情的口吻回答，可以加入一点KD风格的标志性词语（比如“easy money”、“你能防住我？”等调侃感）。
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

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: retrieve_context(x["question"]),
            chat_history=lambda x: get_chat_history(x)
        )
        | prompt
        | llm
        | output_parser
    )
    return chain

# ===== 5. Streamlit 主界面 =====
def main():
    st.set_page_config(page_title="KD AI - 凯文·杜兰特专属助手", page_icon="🏀", layout="wide")

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
        st.markdown("**🏆 关于KD AI**")
        st.info("本助手基于 LangChain + RAG + 记忆，专门回答关于凯文·杜兰特的一切。")

    st.title("🏀 凯文·杜兰特 AI 对话助手")
    st.caption("聊聊篮球，致敬KD — 你可以问关于死神杜兰特的技术、故事、荣誉，甚至模拟和KD对话！")

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

    if prompt := st.chat_input("请输入关于凯文·杜兰特的问题，例如：杜兰特的中投为什么无解？"):
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
