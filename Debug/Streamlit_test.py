import streamlit as st
from Agent_test import InteractiveAgent
#运行方法;
#cd demo\back-end-python\chunkit_fronted到这个目录
#然后Streamlit run Streamlit_test.py可以运行此功能


# 设置页面标题和图标
st.set_page_config(page_title="多智能体系统测试", page_icon="🧪")


# --- 1. 初始化智能体 ---
# 使用 st.cache_resource 确保模型和类只被初始化一次，提高应用性能
@st.cache_resource
def load_agent():
    """
    加载并初始化 InteractiveAgent。
    此函数的结果将被缓存，避免每次页面刷新时都重新加载模型。
    """
    with st.spinner("正在初始化智能体，请稍候..."):
        try:
            agent = InteractiveAgent()
            return agent
        except Exception as e:
            # 如果初始化失败，显示错误并停止应用
            st.error(f"智能体初始化失败: {e}")
            st.stop()


# 加载智能体实例
agent = load_agent()

# --- 2. 侧边栏和模式选择 ---
st.sidebar.title("🛠️ 测试控制台")
st.sidebar.markdown("选择一个接口进行测试，或在主窗口直接开始对话。")

test_mode = st.sidebar.radio(
    "选择测试模式",
    ("完整聊天 (流式)", "仅意图识别")
)

# --- 3. 根据不同模式显示不同界面 ---

# 模式一：完整聊天
if test_mode == "完整聊天 (流式)":
    st.title("🤖 多智能体聊天系统")
    st.caption("这是一个用于测试多智能体 RAG 系统的交互界面。")

    # 初始化聊天记录
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 接收用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 将用户消息添加到历史记录并显示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 获取并显示助手回答
        with st.chat_message("assistant"):

            full_response = ""

            try:
                # 调用核心方法，获取段落生成器
                stream_generator = agent.process_question_with_full_response(prompt, stream_mode=True)

                # 遍历生成器，处理每个返回的段落
                for chunk in stream_generator:
                    if chunk.get("type") == "content":
                        # 从 chunk 中获取头像和段落内容
                        avatar = chunk.get("avatar", "🤖")
                        paragraph = chunk.get("delta", "")

                        if paragraph:
                            # 1. 按照你要求的格式构建输出行
                            output_line = f"头像: {avatar} | 回答段落: {paragraph}"

                            # 2. 使用 st.markdown 直接显示这一行
                            st.markdown(output_line)

                            # 3. 将生成的行添加到完整回复中，用于历史记录
                            full_response += output_line + "\n"

                    elif chunk.get("type") == "error":
                        error_message = chunk.get("message", "未知错误")
                        st.error(f"处理时发生错误: {error_message}")
                        full_response += f"\n\n**错误**: {error_message}"

            except Exception as e:
                st.error(f"调用智能体时发生严重错误: {e}")
                full_response = "抱歉，处理您的请求时发生了严重错误。"
                st.markdown(full_response)
            # --- 核心逻辑修改结束 ---

        # 将助手的完整回答添加到历史记录
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# 模式二：仅意图识别
elif test_mode == "仅意图识别":
    st.title("🎯 意图识别接口测试")
    st.info("在这里，你可以输入问题，系统将只调用 `predict_intent_only` 方法并返回识别出的意图。")

    user_input = st.text_area("输入要识别的问题:", height=100,
                              placeholder="例如：我想咨询心理方面的问题，并了解一下校园图书馆的开放时间。")

    if st.button("识别意图", use_container_width=True):
        if user_input:
            with st.spinner("正在识别..."):
                result = agent.predict_intent_only(user_input)

                st.subheader("识别结果 (原始JSON):")
                st.json(result)

                st.subheader("格式化展示:")
                if result.get("success") and result.get("results"):
                    for intent_info in result["results"]:
                        st.success(
                            f"**意图:** {intent_info.get('intent', 'N/A')} "
                            f"| **头像:** {intent_info.get('avatar', 'N/A')}"
                        )
                else:
                    st.warning(f"未能成功识别意图。消息: {result.get('message', '无')}")
        else:
            st.warning("请输入问题后再进行识别。")