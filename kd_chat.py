# -*- coding: utf-8 -*-
import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough

# ===== 1. 自定义输出解释器（继承 StrOutputParser 并添加风格）=====
class KDStyleOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "抱歉，我现在无法思考清楚。请再问一次！🏀"
        return f"🏀 **KD** 说道：\n\n{cleaned}\n\n---\n*#EasyMoneySniper* 🎯"

# ===== 2. 构建 Chain（提示词模板 + 记忆 + 解析器）=====
def build_chain(memory, output_parser, llm):
    # 提示词模板，让 AI 扮演杜兰特
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位精通篮球的AI助手，专门以篮球巨星凯文·杜兰特（Kevin Durant）的身份或视角回答问题。用自然、热情的口吻回答，可以加入一点KD风格的标志性词语（比如“easy money”、“你能防住我？”等调侃感）。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    def get_chat_history(_inputs):
        # 从 memory 加载历史消息
        if memory and hasattr(memory, "load_memory_variables"):
            mem_vars = memory.load_memory_variables({})
            if "history" in mem_vars:
                return mem_vars["history"]
        return []
    
    chain = (
        RunnablePassthrough.assign(
            history=lambda x: get_chat_history(x)
        )
        | prompt
        | llm
        | output_parser
    )
    return chain

# ===== 3. Streamlit 主界面 =====
def main():
    st.set_page_config(page_title="KD AI - 凯文·杜兰特专属助手", page_icon="🏀", layout="wide")

    with st.sidebar:
        st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/201142.png", width=100, caption="Kevin Durant")
        st.title("⚙️ 模型设置")
        api_key = st.text_input("DeepSeek API Key", type="password",
                                placeholder="不填则读取环境变量 DEEPSEEK_API_KEY")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model_name = st.text_input("模型名称", value="deepseek-chat")
        base_url = st.text_input("API Base URL", placeholder="https://api.deepseek.com/v1",
                                 value="https://api.deepseek.com/v1")
        temperature = st.slider("temperature", 0.0, 1.5, 0.7)
        if st.button("🧹 清空当前对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
            st.rerun()
        st.divider()
        st.markdown("**🏆 关于KD AI**")
        st.info("本助手基于 LangChain + 对话记忆，专门回答关于凯文·杜兰特的一切。")

    st.title("🏀 凯文·杜兰特 AI 对话助手")
    st.caption("聊聊篮球，致敬KD — 你可以问关于死神杜兰特的技术、故事、荣誉，甚至模拟和KD对话！")

    # 初始化 session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)

    if not api_key:
        st.warning("⚠️ 请在上方侧边栏输入 DeepSeek API Key 以开始对话。")
        st.stop()

    # 初始化 LLM 和 Chain
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )
    output_parser = KDStyleOutputParser()
    chain = build_chain(st.session_state.memory, output_parser, llm)

    # 显示历史消息
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🏀"):
                st.markdown(msg["content"])

    # 用户输入
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
                    # 更新 memory
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
