#----文档 处理和向量化的主要逻辑----#

import tiktoken
import numpy as np
import os  # 新增：用于文件路径处理
import torch
import PyPDF2  # 处理PDF
from pathlib import Path
from docx import Document  # 处理Word
from faiss_store_y import FAISSVectorStore  # 引入FAISS向量存储
from sentence_transformers import SentenceTransformer  # 需确保transformers>=4.51.0和sentence-transformers>=2.7.0
from textsplitters import RecursiveCharacterTextSplitter

#从Path文件里面引入知识库文件地址,索引文件的地址
from Utils.Path import (
    PAPER_DOCS_DIR, CAMPUS_DOCS_DIR, FITNESS_DOCS_DIR, PSYCHOLOGY_DOCS_DIR,
    PAPER_INDEX_DIR, CAMPUS_INDEX_DIR, FITNESS_INDEX_DIR, PSYCHOLOGY_INDEX_DIR
)

# 定义所有通用的辅助函数，这些函数不依赖于具体的类
# 注意：函数名统一为不带下划线的，方便直接调用
def read_file(filename: str) -> str:
    """读取文件内容，支持多种编码格式
    txt,md,pdf,docx,
    """
    if filename.endswith('.txt'):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(filename, 'r', encoding='gbk') as file:
                    return file.read()
            except UnicodeDecodeError:
                try:
                    with open(filename, 'r', encoding='ansi') as file:
                        return file.read()
                except Exception as e:
                    raise Exception(f"读取txt文件出错（编码问题）: {str(e)}")
        except Exception as e:
            raise Exception(f"读取txt文件出错: {str(e)}")

    elif filename.endswith('.md') or filename.endswith('.markdown'):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(filename, 'r', encoding='gbk') as file:
                    return file.read()
            except Exception as e:
                raise Exception(f"读取md文件出错: {str(e)}")
        except Exception as e:
            raise Exception(f"读取md文件出错: {str(e)}")

    elif filename.endswith('.pdf'):
        try:
            text = ""
            with open(filename, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            raise Exception(f"读取pdf文件出错: {str(e)}")

    elif filename.endswith('.docx'):
        try:
            doc = Document(filename)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise Exception(f"读取docx文件出错: {str(e)}")

    else:
        raise ValueError(f"不支持的文件格式: {filename}")

def token_length_function(text: str) -> int:
    encoding = tiktoken.get_encoding('cl100k_base')
    return len(encoding.encode(text))

def split_document(filename: str) -> list[str]:
    content = read_file(filename)
    if not content.strip():
        raise ValueError("文件内容为空，无法进行分割")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=token_length_function,
        separators=None,
    )
    chunks = text_splitter.split_text(content)

    if not chunks:
        raise ValueError("文档分割后未得到任何片段，请检查文档内容")

    print(f"文档 {filename} 分割完成，共得到 {len(chunks)} 个片段")
    return chunks

# 抽象父类：封装所有通用的逻辑
class BaseAgentKnowledgeBase:
    """
    所有智能体知识库的抽象基类。
    它封装了文档处理、向量化和存储的通用逻辑。
    子类只需定义各自的配置参数即可。
    """

    #配置各种参数
    def __init__(self, index_path: str, collection_name: str, dimension: int = 1024):
        self.index_path = index_path
        self.collection_name = collection_name
        self.dimension = dimension
        # 在这里调用内部方法
        self.vector_store = self._init_faiss_store(reset=False)
        self.model = self._load_embedding_model()

    def _init_faiss_store(self, reset: bool = False):
        """初始化FAISS向量存储"""
        try:
            vector_store = FAISSVectorStore(
                index_path=self.index_path,
                collection_name=self.collection_name,
                dimension=self.dimension,
                reset=reset
            )
            print(f"初始化FAISS向量存储成功，集合名: {self.collection_name}，当前文档数量: {vector_store.count()}")
            return vector_store
        except Exception as e:
            raise Exception(f"初始化FAISS向量存储失败: {str(e)}")
    def _load_embedding_model(self):
        """加载嵌入模型"""

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # 使用相对路径
            model_path = "Qwen3-Embedding-0___6B"
            
            # 转换为绝对路径并检查是否存在
            abs_path = os.path.abspath(model_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"无法找到模型目录: {abs_path}")
    
            model = SentenceTransformer(
                abs_path,
                device=device,
                tokenizer_kwargs={"padding_side": "left"},
                local_files_only=True  # 强制使用本地文件
            )
            print(f"嵌入模型加载成功，设备: {device}")
            return model
        except Exception as e:
           print(f"加载嵌入模型失败: {str(e)}")
           raise Exception(f"加载嵌入模型失败: {str(e)}")

    def process_folder(self, folder_name: str, reset: bool = False):
        """
        处理指定文件夹及其所有子文件夹中的所有文件，将其分割并存储到数据库。
        """
        if not os.path.isdir(folder_name):
            raise NotADirectoryError(f"文件夹 {folder_name} 不存在或不是一个有效的文件夹")

        # 如果reset=True,则清空数据库
        if reset:
            self.vector_store = self._init_faiss_store(reset=True)

        # 获取所有支持的文件类型
        supported_extensions = {'.txt', '.md', '.markdown', '.pdf', '.docx'}
    
        # 递归获取所有文件
        all_files = []
        for root, dirs, files in os.walk(folder_name):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in supported_extensions:
                    full_path = os.path.join(root, file)
                    all_files.append(full_path)

        if not all_files:
            print(f"文件夹 {folder_name} 及其子文件夹中没有找到任何支持的文件")
            return

        print(f"开始处理文件夹 {folder_name}，共发现 {len(all_files)} 个文件")

        total_chunks = 0
        for file_idx, file_path in enumerate(all_files, 1):
            try:
                filename = os.path.basename(file_path)
                print(f"\n处理第 {file_idx}/{len(all_files)} 个文件: {filename}")
                print(f"文件路径: {file_path}")
            
                # 调用类外的辅助函数
                chunks = split_document(file_path)
                total_chunks += len(chunks)

                embeddings = self.model.encode(chunks)
                documents = []
                embeddings_list = []
                ids = []

                for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    unique_id = f"file_{file_idx}_chunk_{chunk_idx}"
                    documents.append(chunk)
                    embeddings_list.append(embedding.tolist())
                    ids.append(unique_id)

                self.vector_store.add(
                    documents=documents,
                    embeddings=embeddings_list,
                    ids=ids
                )
                print(f"文件 {filename} 处理完成，新增 {len(chunks)} 个片段")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}，将跳过该文件")
                continue

        print(f"\n所有文件处理完成，共处理 {len(all_files)} 个文件，生成 {total_chunks} 个片段")

# 子类：为每个智能体定义具体的知识库
class PsychologyAssistant(BaseAgentKnowledgeBase):
    """心理助手知识库"""

    def __init__(self):
        super().__init__(
            index_path=str(PSYCHOLOGY_INDEX_DIR),
            collection_name="psychology_docs",
            dimension=1024
        )

class CampusQnA(BaseAgentKnowledgeBase):
    """校园知识问答知识库"""

    def __init__(self):
        super().__init__(
            index_path=str(CAMPUS_INDEX_DIR),
            collection_name="campus_docs",
            dimension=1024
        )

class FitnessDietAssistant(BaseAgentKnowledgeBase):
    """健身饮食助手知识库"""

    def __init__(self):
        super().__init__(
            index_path=str(FITNESS_INDEX_DIR),
            collection_name="fitness_docs",
            dimension=1024
        )

class PaperAssistant(BaseAgentKnowledgeBase):
    """论文助手知识库"""

    def __init__(self):
        super().__init__(
            index_path=str(PAPER_INDEX_DIR),
            collection_name="paper_docs",
            dimension=1024
        )


if __name__ == '__main__':
    # 开始处理心理助手知识库
    print("--- 开始处理心理助手知识库 ---")
    psychology_kb = PsychologyAssistant()
    psychology_kb.process_folder(folder_name=str(PSYCHOLOGY_DOCS_DIR), reset=True)

    print("\n" + "=" * 50 + "\n")
    # 开始处理校园知识库
    print("--- 开始处理校园知识库 ---")

    campus_kb = CampusQnA()
    campus_kb.process_folder(folder_name=str(CAMPUS_DOCS_DIR), reset=True)

    print("\n" + "=" * 50 + "\n")
    

    # 开始处理健身饮食助手知识库
    print("--- 开始处理健身饮食助手知识库 ---")
    fitness_kb = FitnessDietAssistant()
    fitness_kb.process_folder(folder_name=str(FITNESS_DOCS_DIR), reset=True)

    print("\n" + "=" * 50 + "\n")

    # 开始处理论文助手知识库
    print("--- 开始处理论文助手知识库 ---")
    paper_kb = PaperAssistant()
    paper_kb.process_folder(folder_name=str(PAPER_DOCS_DIR), reset=True)

    print("\n所有知识库处理完成！")
