# -*- coding: utf-8 -*-
"""
凯文·杜兰特 AI 对话助手
技术栈：提示词模板、输出解释器、Chain链、Memory、RAG
LangChain 0.3.0 + Streamlit
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
# 完整静态知识库（基于提供的最新生平，截至2025年火箭队）
STATIC_KNOWLEDGE = [
    # 1. 基本信息
    "凯文·杜兰特（Kevin Durant），全名凯文·韦恩·杜兰特（Kevin Wayne Durant）。1988年9月29日出生于美国华盛顿哥伦比亚特区，身高2.11米，体重108公斤，司职小前锋。目前效力于NBA休斯敦火箭队。2007年NBA选秀首轮第二顺位被西雅图超音速队选中。",
    
    # 2. 早年经历（童年）
    "杜兰特由母亲旺达·杜兰特和外婆抚养长大，生父在他不到一岁时离开，13岁才重逢。他和哥哥托尼一起长大，童年崇拜文斯·卡特。8岁时遇到篮球导师查尔斯·克雷格，开始系统训练。曾因训练太苦想放弃，被母亲一巴掌打醒，从此坚定信念。",
    
    # 3. 高中与大学
    "高中转学至橡树山高中，场均19.6分8.8篮板，入选全美最佳阵容二队。后转至国家基督学院，在传奇教头斯图·维特指导下进步。2006年进入德克萨斯大学，大一赛季场均近26分11篮板，包揽奈史密斯奖和伍登奖，成为首位获此殊荣的大一新生。德克萨斯大学退役了他的35号球衣。2007年4月宣布参加NBA选秀。",
    
    # 4. 雷霆时期（含超音速）
    "2007年被超音速选中，新秀赛季场均20.3分，当选最佳新秀。2008年球队搬迁至俄克拉荷马城更名为雷霆队，杜兰特与威斯布鲁克、哈登组成‘雷霆三少’。2010年成为史上最年轻得分王，2010-2014年间共获得4次得分王。2013-14赛季荣膺常规赛MVP，场均32.0分7.4篮板5.5助攻。2012年率队打入总决赛，负于热火。在雷霆队总得分17566分，队史第二。",
    
    # 5. 勇士时期
    "2016年夏天以自由球员身份加盟金州勇士队。2016-17、2017-18赛季连续两年夺得NBA总冠军，并两度荣膺总决赛MVP，成为继乔丹和奥尼尔之后第三位蝉联FMVP的球员。2019年总决赛中遭遇跟腱断裂重伤。",
    
    # 6. 篮网时期
    "2019年加盟布鲁克林篮网队，因跟腱伤势首个赛季报销。2021年入选NBA75大巨星。与欧文、哈登组成‘篮网三巨头’，但因伤病未能夺冠。",
    
    # 7. 太阳时期
    "2023年交易截止日被交易至菲尼克斯太阳队，太阳送出布里奇斯、卡梅隆·约翰逊、4个首轮签和1个首轮签互换权。",
    
    # 8. 火箭时期（最新）
    "2025年6月23日，太阳队与火箭队达成一笔8换1交易，杜兰特被交易至休斯敦火箭队，这是他生涯第4次换队。",
    
    # 9. 生涯数据（截至2025）
    "共出战1061场比赛（1058场首发），场均27.3分、7.0篮板、4.4助攻、1.1抢断、1.1盖帽，投篮命中率50.1%，三分命中率38.7%，罚球命中率88.4%。生涯总得分28,924分（历史第8），罚球命中数6,993个（历史第10），进球数9,950个（历史第17）。是NBA历史上仅有的两位生涯场均25+且真实命中率不低于60%的球员之一，也是唯一在至少1000场比赛中投篮50%+、三分35%+、罚球85%+的球员。2012-13赛季加入50/40/90俱乐部；2022-23赛季成为史上首位单赛季（出场超半数）投篮55%+、三分40%+、罚球90%+的球员。",
    
    # 10. 主要荣誉
    "主要荣誉：2次NBA总冠军（2017、2018）、2次总决赛MVP、1次常规赛MVP（2013-14）、4次得分王、14次全明星、6次最佳阵容一阵、4次最佳阵容二阵、2次全明星MVP、1次最佳新秀。",
    
    # 11. 国家队生涯
    "美国男篮历史上最伟大的球员之一，被誉为‘美国男篮GOAT’。拥有4枚奥运会金牌（2012、2016、2020、2024）和1枚世锦赛金牌（2010）。荣获2020年东京奥运会MVP和2010年世锦赛MVP。是奥运男篮史上首位4金王。曾3次当选美国篮球年度最佳男运动员（2010、2016、2021）。",
    
    # 12. 个人生活与家庭
    "第一个公开女友是WNBA球员莫妮卡·赖特，2013年求婚成功。与母亲旺达关系亲密。35号球衣号码是为了纪念篮球启蒙导师查尔斯·克雷格（35岁时被谋杀）。",
    
    # 13. 商业与投资
    "2007年与耐克签订7年6000万美元代言合同（当时新秀鞋合同历史第二高）。与里奇·克莱曼共同创立杜兰特公司、Thirty Five Ventures和Boardroom。投资了Postmates、Acorns等科技公司，投资酒店业，并通过Boardroom持有法甲豪门巴黎圣日耳曼12.5%股份。长期慈善，曾向俄克拉荷马城受灾家庭子女教育基金会捐款100万美元。",
    
    # 14. 技术特点与时间线（合并）
    "技术特点：身高2.11米却拥有后卫般的敏捷和速度，干拔跳投无解，生涯投篮命中率50.1%，三分38.7%，是历史顶级得分手。重要时间线：1988年出生；2006年大学；2007年选秀；2008年最佳新秀；2010年最年轻得分王+世锦赛MVP；2012年首次总决赛+奥运金牌；2014年常规赛MVP；2016年加盟勇士+奥运金牌；2017、2018总冠军+FMVP；2019年加盟篮网+跟腱断裂；2020年奥运金牌+MVP；2021年入选75大；2023年交易至太阳；2024年巴黎奥运第4金；2025年交易至火箭，总得分突破29000分。",
    
    # 15. 名言
    "名言：'Hard work beats talent when talent fails to work hard.'"
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

# ========================= 2. 输出解释器 =========================
class KDStyleOutputParser(StrOutputParser):
    """自定义输出解析器，添加KD风格前缀和标签"""
    def parse(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "抱歉，我现在无法思考清楚。请再问一次！🏀"
        if cleaned.startswith("KD 说道："):
            cleaned = cleaned[7:].strip()
        return f"🏀 **KD** 说道：\n\n{cleaned}\n\n---\n*#EasyMoneySniper* 🎯"

# ========================= 3. 构建 Chain =========================
def build_chain(retriever, memory, output_parser, llm, current_team):
    system_template = """
你以凯文·杜兰特的第一人称回答问题。
请基于以下从知识库中检索到的相关信息（关于你的生涯、荣誉等）：
{context}

【重要】如果你被问到“你现在在哪支球队”或类似问题，请回答：**{current_team}**（这是经过确认的最新事实）。
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
    
    chain = (
        RunnablePassthrough.assign(
            history=get_history,
            context=lambda x: retrieve_context(x["question"]),
            current_team=lambda x: x.get("current_team", current_team)
        )
        | prompt
        | llm
        | output_parser
    )
    return chain

# ========================= 4. Streamlit 界面 =========================
def main():
    st.set_page_config(page_title="KD AI 对话助手", page_icon="🏀", layout="wide")
    
    with st.sidebar:
        st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/201142.png", width=100)
        st.title("⚙️ 模型配置")
        
        api_key = st.text_input("DeepSeek API Key", type="password",
                                placeholder="不填则读取环境变量 DEEPSEEK_API_KEY")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model_name = st.text_input("模型名称", value="deepseek-chat")
        base_url = st.text_input("API Base URL", value="https://api.deepseek.com/v1")
        temperature = st.slider("temperature", 0.0, 1.5, 0.3)
        
        st.divider()
        
        if st.button("🧹 清空对话"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
    
    # 主界面
    st.title("🏀 凯文·杜兰特 AI 助手")
    st.caption("具备提示词模板 | 输出解释器 | Chain链 | 对话记忆 | RAG增强检索")
    
    # 初始化 session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="history")
    # 固定球队为休斯敦火箭队
    CURRENT_TEAM = "休斯敦火箭队"
    if "current_team" not in st.session_state:
        st.session_state.current_team = CURRENT_TEAM
    
    if not api_key:
        st.warning("⚠️ 请在上方侧边栏输入 DeepSeek API Key。")
        st.stop()
    
    st.info(f"📌 当前球队：**{st.session_state.current_team}**")
    
    retriever = get_retriever()
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key, base_url=base_url)
    parser = KDStyleOutputParser()
    chain = build_chain(retriever, st.session_state.memory, parser, llm, st.session_state.current_team)
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏀" if msg["role"]=="assistant" else None):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("请问关于凯文·杜兰特的问题（例如：你现在在哪支球队？）"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🏀"):
            with st.spinner("检索知识库并生成回答..."):
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
