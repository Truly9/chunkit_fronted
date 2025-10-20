import json
import os
import sys
from openai import OpenAI
from collections import Counter

# LangChain 相关库
# 需要安装: pip install langchain langchain-openai faiss-cpu tiktoken
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # 用于本地模型
from langchain.schema import Document

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print("--- 引入qwen3-Embeddings模型 ---")
model_path = os.path.join(parent_dir, "Qwen3-Embedding-0___6B")
embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={"device": "cpu"}
)
print("--- 引入qwen3-Turbo模型 ---")
client = OpenAI(
    api_key="sk-93817db303964020bbc79b017be4768b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=30.0
)



# ---  RAG类,（用语料库）构建的索引 <IntentRecognition\faiss_corpus_index> 用于意图识别，目的是将用户问题分类到四个意图之一  ---

class RagQueryEnhancer:
    def __init__(self, corpus_folder=None, index_path="faiss_corpus_index"):
        """
        构造函数：初始化所有资源，特别是加载或构建向量数据库。
        """
        print("--- 正在初始化 RagQueryEnhancer ---")
        if not embeddings or not client:
            raise RuntimeError("基础模型（Embedding或LLM）未成功加载，无法初始化。")

        # 如果未指定corpus_folder，使用默认路径
        if corpus_folder is None:
            # 计算corpus文件夹的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # corpus在当前文件的上一级目录中
            corpus_folder = os.path.join(os.path.dirname(current_dir), "corpus")
            print(f"使用默认corpus路径: {corpus_folder}")

        # 封装vector_store
        self.vector_store = self._initialize_vector_store(corpus_folder, index_path)

        if not self.vector_store:
            raise RuntimeError("向量数据库初始化失败，增强器无法工作。")

        print("--- RagQueryEnhancer 初始化成功，准备就绪！ ---")
    def _initialize_vector_store(self, corpus_folder, index_path):
        """（内部方法）加载或构建FAISS索引，并将其作为实例变量返回"""
        # 使用绝对路径保存索引
        current_dir = os.path.dirname(os.path.abspath(__file__))
        index_abs_path = os.path.join(current_dir, index_path)
        
        if os.path.exists(index_abs_path):
            print(f"发现本地索引，正在从 '{index_abs_path}' 加载...")
            try:
                return FAISS.load_local(index_abs_path, embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"加载索引失败: {e}。将尝试重新构建。")

        print("未找到或加载失败，开始构建新索引...")
        all_docs = self._load_knowledge_base(corpus_folder)
        if all_docs:
            print("正在构建向量库 (这可能需要一些时间)...")
            try:
                # faiss封装好的 from_documents函数，也就是向量化的步骤
                vs = FAISS.from_documents(all_docs, embeddings)
                print(f"新索引构建成功，正在保存到 '{index_abs_path}'...")
                # faiss封装好的 save函数
                vs.save_local(index_abs_path)
                return vs
            except Exception as e:
                print(f"构建或保存索引时出错: {e}")
        return None

    def _load_knowledge_base(self, corpus_folder):
        """（内部方法）从JSON文件加载知识"""
        intent_mapping = {
            "Campus_2000.json": "校园知识问答助手", "Fitness_2000.json": "健身饮食助手",
            "Paper_2000.json": "论文助手", "Heart_2000.json": "心理助手"
        }
        all_documents = []

        for filename, intent in intent_mapping.items():
            filepath = os.path.join(corpus_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    queries = json.load(f)
                    for query in queries:
                        all_documents.append(Document(page_content=query, metadata={'intent': intent}))
            except Exception as e:
                print(f"警告: 处理文件 '{filepath}' 时出错: {e}")
        print(f"共加载 {len(all_documents)} 个文档。")
        return all_documents

    def enhance_query(self, user_query: str, k: int = 9):
        """
        【核心公共方法】
        执行完整的RAG流程：识别意图并生成一个或多个强化后的查询。
        """
        # --- 1. 检索 ---
        # 直接使用 self.vector_store，它在初始化时就已经准备好了！

        #使用faiss封装好的 similarity_search 函数
        retrieved_docs = self.vector_store.similarity_search(user_query, k=k)
        if not retrieved_docs:
            return {"original_query": user_query, "analysis_results": [], "error": "无法检索到相关文档"}

        # --- 2. 意图识别 ---
        intents = [doc.metadata['intent'] for doc in retrieved_docs]
        intent_counts = Counter(intents).most_common()

        # --- 3. 决定生成策略 (单意图或双意图) ---
        prompts_to_generate = []
        most_common_intent, most_common_count = intent_counts[0]

        if most_common_count >= 5:  # 高置信度场景
            prompts_to_generate.append({"intent": most_common_intent,
                                        "prompt": self._generate_rewrite_prompt(most_common_intent, retrieved_docs,
                                                                                user_query)})
        else:  # 低置信度场景
            for intent, count in intent_counts[:2]:  # 取前两个
                prompts_to_generate.append(
                    {"intent": intent, "prompt": self._generate_rewrite_prompt(intent, retrieved_docs, user_query)})

        # --- 4. 调用LLM生成 ---
        final_output = []
        for task in prompts_to_generate:
            try:
                response = client.chat.completions.create(
                    model="qwen-turbo",
                    messages=[{"role": "user", "content": task["prompt"]}],
                    temperature=0.2,
                )
                rewritten_query = response.choices[0].message.content.strip()
                final_output.append({"intent": task["intent"], "rewritten_query": rewritten_query})
            except Exception as e:
                final_output.append({"intent": task["intent"], "error": str(e)})

                # --- 5. 【修改】返回包含意图分布的完整结果 ---
        return {
                "original_query": user_query,
                "analysis_results": final_output,
                "intent_distribution": dict(intent_counts)  # 将Counter转为普通字典
            }


    def _generate_rewrite_prompt(self,target_intent, all_docs,user_query):
        relevant_examples = [doc.page_content for doc in all_docs if doc.metadata['intent'] == target_intent]
        examples_str = "\n".join([f"- {ex}" for ex in relevant_examples])

        return f"""你是一个Prompt工程专家，任务是优化用户的提问。

    背景信息:
    用户的意图似乎是关于“{target_intent}”。
    与这个领域相关的其他用户提问有：
    {examples_str}

    你的任务:
    请你结合以上背景信息，将用户的“原始问题”改写成一个更具体、更结构化、更符合其“{target_intent}”需求的专业级提问。
    这个新的专业级提问将交给另一个专家模型来回答，所以它需要清晰、信息量丰富。
    请直接输出改写后的新提问，不要包含任何前缀或解释。

    原始问题:
    {user_query}

    改写后的专业级提问:
    """


# --- 用于独立测试此文件的代码块 ---
if __name__ == '__main__':
    print("\n=== 正在独立运行 Intent_by_Rag.py 进行测试 ===")
    try:
        enhancer = RagQueryEnhancer()
        test_query = "写论文压力好大，睡不着觉，有什么锻炼可以缓解吗？"
        result = enhancer.enhance_query(test_query)
        print("\n--- 测试结果 ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")