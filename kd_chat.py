import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

# 输出解释器
class KDStyleOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "抱歉，我现在无法思考清楚。请再问一次！🏀"
        return f"🏀 **KD** 说道：\n\n{cleaned}\n\n---\n*#EasyMoneySniper* 🎯"

# 构建 RAG 向量库（使用 DeepSeek Embedding）
def build_kd_vectorstore(api_key: str, base_url: str):
    kd_docs = [
        "凯文·杜兰特 (Kevin Durant) 1988年9月29日出生于美国华盛顿特区，绰号'KD'、'死神'。",
        "杜兰特在2007年NBA选秀中以榜眼身份被西雅图超音速队选中，新秀赛季获得最佳新秀。",
        "2014年，杜兰特荣获NBA常规赛最有价值球员(MVP)，他的MVP演讲'你才是真正的MVP'感动无数人。",
        "杜兰特职业生涯4次获得NBA得分王，被誉为历史顶级得分手，拥有无解的中距离和三分能力。",
        "2016年杜兰特加盟金州勇士队，连续两年（2017、2018）夺得NBA总冠军并获得总决赛MVP。",
        "2019年总决赛期间杜兰特遭遇跟腱断裂重伤，但康复后依然保持巅峰状态，展现了坚强的意志。",
        "2021年，杜兰特代表美国男篮夺得东京奥运会金牌，成为美国队史奥运得分王。",
        "杜兰特生涯荣誉：2次总冠军、2次FMVP、1次MVP、4次得分王、13次全明星、10次最佳阵容。",
        "杜兰特身高208cm，臂展225cm，招牌干拔跳投和变向突破。",
        "杜兰特经典名言：'Hard work beats talent when talent fails to work hard.'",
        "2023年杜兰特被交易至菲尼克斯太阳队，与布克、比尔组成三巨头。",
        "杜兰特职业生涯总得分历史前十，季后赛总得分也名列前茅。",
        "帮助美国队夺得4枚奥运金牌。"
    ]
    documents = [Document(page_content=doc) for doc in kd_docs]
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    # 关键：使用 deepseek-embed 模型名
    embeddings = OpenAIEmbeddings(
        model="deepseek-embed",
        openai_api_key=api_key,
        openai_api_base=base_url
    )
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def get_llm(api_key: str, model_name: str, temperature: float, base_url: str):
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )

def build_chain(retriever, memory, output_parser, llm):
    system_template = """
你是一位精通篮球的AI助手，专门以篮球巨星凯文·杜兰特（Kevin Durant）的身份或视角回答问题。
请严格基于以下【背景知识】进行回答，如果背景知识不足，可以结合你自己的篮球知识。

【背景知识】
{context}

用户问题：{question}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    def get_chat_history(_inputs):
        if memory and hasattr(memory, "load_memory_variables"):
            mem_vars = memory.load_memory_variables({})
            if "history" in mem_vars:
                return mem_vars["history"]
        return []

    def retrieve_context(question):
        if retriever:
            docs = retriever.invoke(question)
            return "\n\n".join([doc.page_content for doc in docs])
        return "暂无检索到的背景知识。"

    chain = (
        RunnablePassthrough.assign(
            history=lambda x: get_chat_history(x),
            context=lambda x: retrieve_context(x["question"])
        )
        | prompt
        | llm
        | output_parser
    )
    return chain

def main():
    st.set_page_config(page_title="KD AI RAG版", page_icon="🏀", layout="wide")

    with st.sidebar:
        st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/201142.png", width=100, caption="Kevin Durant")
        st.title("⚙️ 模型设置")
        api_key = st.text_input("DeepSeek API Key", type="password")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model_name = st.text_input("模型名称", value="deepseek-chat")
        base_url = st.text_input("API Base URL", value="https://api.deepseek.com/v1")
        temperature = st.slider("temperature", 0.0, 1.5, 0.7)
        if st.button("🧹 清空对话"):
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
            st.rerun()
        st.info("基于 RAG (DeepSeek Embedding) + 记忆")

    st.title("🏀 凯文·杜兰特 AI 助手（RAG增强）")
    st.caption("知识库包含杜兰特生涯数据，可精确回答历史问题")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    if not api_key:
        st.warning("请输入 DeepSeek API Key")
        st.stop()

    if st.session_state.retriever is None:
        with st.spinner("初始化向量库 (使用 deepseek-embed)..."):
            try:
                st.session_state.retriever = build_kd_vectorstore(api_key, base_url)
                st.success("知识库加载完成")
            except Exception as e:
                st.error(f"初始化失败: {e}\n请确认 API Key 有效且支持 embedding")
                st.stop()

    llm = get_llm(api_key, model_name, temperature, base_url)
    parser = KDStyleOutputParser()
    chain = build_chain(st.session_state.retriever, st.session_state.memory, parser, llm)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏀" if msg["role"]=="assistant" else None):
            st.markdown(msg["content"])

    if prompt := st.chat_input("提问："):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🏀"):
            with st.spinner("检索知识库..."):
                try:
                    response = chain.invoke({"question": prompt})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(response)
                except Exception as e:
                    error_msg = f"错误: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(error_msg)

if __name__ == "__main__":
    main()
