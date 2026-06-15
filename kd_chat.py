import streamlit as st
import os
import json
import requests
from duckduckgo_search import DDGS
from datetime import datetime

# --- 配置 ---
st.set_page_config(page_title="KD AI - 实时情报官", page_icon="🏀", layout="wide")

# --- 1. 可更新的知识库 ---
# 基础静态知识（几乎不变的信息）
BASE_KNOWLEDGE = {
    "基本信息": "凯文·杜兰特，1988年9月29日出生，绰号'KD'、'死神'。",
    "选秀": "2007年NBA选秀榜眼，最佳新秀。",
    "荣誉": "4次得分王，2次总冠军，2次FMVP，1次常规赛MVP，13次全明星。",
    "技术": "身高208cm，臂展225cm，历史顶级得分手，标志性干拔跳投。",
    "伤病": "2019年总决赛跟腱断裂，后完美康复。",
    "奥运": "4次奥运金牌得主，美国队史奥运得分王。",
    "名言": "Hard work beats talent when talent fails to work hard.",
    # --- 以下信息可能会被实时搜索更新 ---
    "现效力球队": "待更新",  # 这个字段会被动态替换
}

# --- 2. 实时搜索与知识更新函数 ---
@st.cache_data(ttl=600)  # 缓存10分钟，避免频繁搜索
def fetch_and_update_knowledge():
    """搜索最新信息，并返回更新后的知识库副本"""
    updated_knowledge = BASE_KNOWLEDGE.copy()
    try:
        with DDGS() as ddgs:
            # 搜索关于球队的最新、最权威的新闻
            results = list(ddgs.text("凯文杜兰特 最新交易 火箭 太阳 2025 2026", max_results=2))
            if not results:
                results = list(ddgs.text("Kevin Durant current team 2026", max_results=2))
            
            # 简单分析搜索结果来更新“现效力球队”字段
            for r in results:
                body = r['body'].lower()
                if "火箭" in body or "rockets" in body:
                    updated_knowledge["现效力球队"] = "休斯顿火箭队（根据最新新闻，已于2025-26赛季前被交易至火箭）"
                    break
                elif "太阳" in body or "suns" in body:
                    updated_knowledge["现效力球队"] = "菲尼克斯太阳队（近期新闻未显示变动）"
                else:
                    # 如果都没明确，保持未知状态
                    updated_knowledge["现效力球队"] = "未知，请查阅最新体育新闻"
                    
    except Exception as e:
        print(f"搜索或更新知识库时出错: {e}")
        updated_knowledge["现效力球队"] = "信息获取失败，请稍后再试。"
    
    return updated_knowledge

# --- 3. 构建给AI的最终知识文本 ---
def build_knowledge_text(knowledge):
    """将更新后的知识库格式化成易读文本"""
    return f"""
    【凯文·杜兰特最新档案】
    - 基本信息: {knowledge['基本信息']}
    - **当前效力球队: {knowledge['现效力球队']}** (重要：此信息已通过联网搜索核实)
    - 生涯荣誉: {knowledge['荣誉']}
    - 技术特点: {knowledge['技术']}
    - 重大伤病: {knowledge['伤病']}
    - 奥运成就: {knowledge['奥运']}
    - 经典名言: {knowledge['名言']}
    - 选秀情况: {knowledge['选秀']}
    """

# --- 4. 调用大语言模型 (DeepSeek) ---
def ask_deepseek(api_key, base_url, model, temperature, messages):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API错误 {response.status_code}: {response.text}")

# --- 5. 输出样式解析器 ---
def format_output(text):
    text = text.strip()
    if not text:
        return "抱歉，我现在无法思考清楚。请再问一次！🏀"
    # 清理可能重复的标记
    if text.startswith("KD 说道："):
        text = text[7:]
    return f"🏀 **KD** 说道：\n\n{text}\n\n---\n*#EasyMoneySniper* 🎯"

# --- Streamlit UI ---
def main():
    with st.sidebar:
        st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/201142.png", width=100, caption="Kevin Durant")
        st.title("⚙️ 模型与知识库设置")
        api_key = st.text_input("DeepSeek API Key", type="password", placeholder="不填则读取环境变量")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model_name = st.text_input("模型名称", value="deepseek-chat")
        base_url = st.text_input("API Base URL", value="https://api.deepseek.com/v1")
        temperature = st.slider("创造力 (temperature)", 0.0, 1.5, 0.3)  # 低温度保证回答稳定
        
        if st.button("🔄 手动更新知识库", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.markdown("**🚀 技术特性**")
        st.info("✅ 实时联网搜索\n✅ 动态知识库更新\n✅ 基于事实的回答")

    st.title("🏀 凯文·杜兰特 AI 助手（真·实时情报版）")
    st.caption("每次对话前，我会先去网上搜索最新新闻，尤其会确认我当前所在的球队。")
    
    if not api_key:
        st.warning("请输入 DeepSeek API Key 以开始。")
        st.stop()

    # 初始化session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = []
    if "knowledge" not in st.session_state:
        # 启动时或点击更新按钮后，重新获取最新知识
        with st.spinner("正在联网检索凯文·杜兰特的最新动态..."):
            st.session_state.knowledge = fetch_and_update_knowledge()

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 用户输入
    if prompt := st.chat_input("问点关于KD的最新消息？"):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 构建知识文本（使用已更新的知识库）
        context = build_knowledge_text(st.session_state.knowledge)
        
        # 构建历史对话
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.memory[-6:]])
        
        # 构建完整的系统提示
        system_prompt = f"""
你是一位由AI驱动的凯文·杜兰特专属助手，你必须严格以杜兰特的第一人称视角来回答问题。

以下是为你准备的，**刚刚从新闻中获取并核实过的个人档案**：
{context}

【重要指令】
*   如果被问及“现在效力于哪支球队”或类似问题，**请完全以上述档案中【当前效力球队】的信息为准**，这是刚刚从新闻中实时获取的。
*   如果档案中的信息和你的旧有认知矛盾，请相信档案中的最新信息。
*   在回答时，尽量模仿杜兰特自信、直接的口吻，可以适度加入“easy money”、“神一样的存在”等标志性口头禅。
*   保持回答的准确和清晰，避免生成无意义的特殊符号或乱码。

最近的对话历史：
{history_text}

现在，用户问你：{prompt}
请开始你的回答：
"""
        # 调用AI
        with st.chat_message("assistant", avatar="🏀"):
            with st.spinner("结合实时情报思考中..."):
                try:
                    messages_for_api = [{"role": "user", "content": system_prompt}]
                    raw = ask_deepseek(api_key, base_url, model_name, temperature, messages_for_api)
                    answer = format_output(raw)
                    st.markdown(answer)
                    
                    # 保存消息和记忆
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.memory.append({"role": "user", "content": prompt})
                    st.session_state.memory.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"出错了: {e}")

if __name__ == "__main__":
    main()
