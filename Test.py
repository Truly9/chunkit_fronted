r"""
Test.py
一体化测试脚本：使用Streamlit实现整个项目的功能。
- 知识库处理：支持校园领域的文字+图片处理（MultiRAG），其他领域文字处理（builder）。
- 问答反馈：检索展示匹配的文字与图片路径 + 流式问答输出。
- 统一Web界面：选择领域、输入问题、显示答案。

使用示例：
1）启动Streamlit应用：
   streamlit run Test.py
"""

import streamlit as st
import sys
import time
import re
from typing import List, Dict, Any

# 项目内模块导入
from Utils.Path import (
    PAPER_DOCS_DIR, CAMPUS_DOCS_DIR, FITNESS_DOCS_DIR, PSYCHOLOGY_DOCS_DIR,
    PAPER_INDEX_DIR, CAMPUS_INDEX_DIR, FITNESS_INDEX_DIR, PSYCHOLOGY_INDEX_DIR,
    CAMPUS_IMAGES_PATH, CAMPUS_IMAGES_MAPPING_PATH,
)

from multiRAG import MultiRAG
from ClassAssistant.callback import BaseAssistant as CampusLLM  # 校园域：使用MultiRAG + LLM流式回答（含图片信息）

from ClassAssistant.RAGlibrary import (
    RAG_psychology,
    RAG_fitness,
    RAG_compus,
    RAG_paper,
    RAG
)

from retrieve_model import retrieve_relevant_chunks

# ----------------------------
# 工具函数
# ----------------------------

def parse_image_paths_from_text(text: str) -> List[str]:
    r"""从流式段落文本中解析出图片路径（兼容两种提示格式）。
    兼容：[图片地址: D:\...] 或 [地址: D:\...]
    """
    paths = []
    patterns = [r"\[图片地址:\s*([^\]]+)\]", r"\[地址:\s*([^\]]+)\]"]
    for pat in patterns:
        for m in re.finditer(pat, text):
            path = m.group(1).strip()
            if path:
                paths.append(path)
    return paths


def format_matches(results: List[Dict[str, Any]]) -> str:
    """格式化 MultiRAG.retrieve 的匹配结果（含文本与图片）。"""
    if not results:
        return "未检索到相关内容。"
    
    output = f"检索到 {len(results)} 个相关结果：\n\n"
    for i, r in enumerate(results, 1):
        rtype = r.get('type', 0)
        content = r.get('document', '')
        source = r.get('source', '')
        label = '图片' if rtype == 1 else '文字'
        output += f"—— 结果 {i}（{label}）——\n"
        output += f"{content[:300]}\n"
        if rtype == 1 and source:
            output += f"图片路径: {source}\n"
        output += "\n"
    
    return output


# ----------------------------
# Streamlit应用
# ----------------------------

def main():
    st.set_page_config(page_title="RAG问答工具", layout="wide")
    st.title("RAG 问答工具")
    
    # 侧边栏设置
    st.sidebar.title("设置")
    domain = st.sidebar.selectbox(
        "选择领域",
        ["campus", "paper", "fitness", "psychology"],
        format_func=lambda x: {
            "campus": "校园",
            "paper": "论文",
            "fitness": "健身",
            "psychology": "心理"
        }.get(x, x)
    )
    
    topk = st.sidebar.slider("检索结果数量", 1, 10, 5)
    show_sources = st.sidebar.checkbox("显示检索源", True)
    
    # 初始化RAG（根据所选领域）
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False
        st.session_state.rag_instances = {}
    
    # 用户输入
    query = st.text_input("请输入你的问题：")
    
    if st.button("提交查询") and query:
        with st.spinner(f"正在处理您关于{domain}领域的问题..."):
            # 根据领域选择不同的处理方式
            if domain == "campus":
                # 检索（含图片路径）
                rag = MultiRAG(
                    index_path=str(CAMPUS_INDEX_DIR),
                    collection_name='campus_docs',
                    image_output_dir=str(CAMPUS_IMAGES_PATH),
                    image_mapping_file=str(CAMPUS_IMAGES_MAPPING_PATH),
                )
                results = rag.retrieve(query, topk=topk)
                
                if show_sources:
                    st.subheader("检索结果")
                    st.text(format_matches(results))
                
                # 流式回答
                st.subheader("回答")
                answer_container = st.empty()
                llm = CampusLLM()
                llm.start_LLM()
                full_text = ""
                candidate_image_paths = []
                
                try:
                    for para in llm.retrieve_and_answer(query, top_k=max(5, topk)):
                        full_text += para + "\n"
                        answer_container.markdown(full_text)
                        # 解析图片路径
                        paths = parse_image_paths_from_text(para)
                        if paths:
                            candidate_image_paths.extend(paths)
                finally:
                    llm.close_LLM()
                
                # 显示图片（如有）
                if candidate_image_paths:
                    st.subheader("相关图片")
                    for i, path in enumerate(candidate_image_paths):
                        st.image(path, caption=f"图片 {i+1}")
            
            elif domain in ["paper", "fitness", "psychology"]:
                # 使用对应领域的RAG实例
                if domain not in st.session_state.rag_instances:
                    if domain == "paper":
                        st.session_state.rag_instances[domain] = RAG_paper()
                    elif domain == "fitness":
                        st.session_state.rag_instances[domain] = RAG_fitness()
                    elif domain == "psychology":
                        st.session_state.rag_instances[domain] = RAG_psychology()
                
                rag_instance = st.session_state.rag_instances[domain]
                
                # 检索相关内容
                if show_sources:
                    index_path_map = {
                        "paper": PAPER_INDEX_DIR,
                        "fitness": FITNESS_INDEX_DIR,
                        "psychology": PSYCHOLOGY_INDEX_DIR
                    }
                    
                    chunks = retrieve_relevant_chunks(
                        query, 
                        index_path=str(index_path_map[domain]),
                        top_k=topk
                    )
                    
                    if chunks:
                        st.subheader("检索结果")
                        for i, chunk in enumerate(chunks, 1):
                            st.text(f"—— 结果 {i} ——\n{chunk[:300]}")
                
                # 流式回答
                st.subheader("回答")
                answer_container = st.empty()
                answer = ""
                
                for delta in rag_instance.call_RAG(query):
                    answer += delta
                    answer_container.markdown(answer)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"运行出错: {e}")
        sys.exit(1)