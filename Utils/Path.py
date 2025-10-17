"""
path.py
该模块集中管理项目的所有路径，包括知识库文件、索引文件等。
"""

from pathlib import Path

# ==============================
# --- 项目根目录 ---
# ==============================
# 将项目根目录定义为 chunkit_fronted 文件夹的父目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==============================
# --- 知识库目录 (Knowledge Base) ---
# ==============================
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "Knowledge_Base"

# 各智能体的知识库文件夹
PAPER_DOCS_DIR = KNOWLEDGE_BASE_DIR / "paper_docs"
CAMPUS_DOCS_DIR = KNOWLEDGE_BASE_DIR / "campus_docs"
FITNESS_DOCS_DIR = KNOWLEDGE_BASE_DIR / "fitness_docs"
PSYCHOLOGY_DOCS_DIR = KNOWLEDGE_BASE_DIR / "psychology_docs"

# ==============================
# --- 向量索引目录 (FAISS Index) ---
# ==============================
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"

# 各智能体的向量索引文件夹
PAPER_INDEX_DIR = FAISS_INDEX_DIR / "paper"
CAMPUS_INDEX_DIR = FAISS_INDEX_DIR / "campus"
FITNESS_INDEX_DIR = FAISS_INDEX_DIR / "fitness"
PSYCHOLOGY_INDEX_DIR = FAISS_INDEX_DIR / "psychology"

# ==============================
# ---图片处理后的根目录---
# ==============================
ALL_PROCESSED_IMAGES_DIR = PROJECT_ROOT / "All_Processed_Images"

# 各智能体图片处理后的文件夹目录
CAMPUS_IMAGES_DIR = ALL_PROCESSED_IMAGES_DIR / "campus"
PAPER_IMAGES_DIR = ALL_PROCESSED_IMAGES_DIR / "paper"
FITNESS_IMAGES_DIR = ALL_PROCESSED_IMAGES_DIR / "fitness"
PSYCHOLOGY_IMAGES_DIR = ALL_PROCESSED_IMAGES_DIR / "psychology"

# 各个智能体图片处理后提取图片的文件路径
CAMPUS_PROCESSED_EXTRACTED_IMAGES = CAMPUS_IMAGES_DIR / "campus_extracted_images.docx"
PSYCHOLOGY_PROCESSED_EXTRACTED_IMAGES = PSYCHOLOGY_IMAGES_DIR / "psychology_extracted_images.docx"
# 提取图片后转的json文件路径
CAMPUS_EXTRACTED_IMAGES_JSON = CAMPUS_IMAGES_DIR / "campus_extracted_images.json"
PSYCHOLOGY_EXTRACTED_IMAGES_JSON = PSYCHOLOGY_IMAGES_DIR / "psychology_extracted_images.json"
# 图片映射文件路径
CAMPUS_IMAGES_MAPPING_PATH = CAMPUS_IMAGES_DIR / "campus_images_mapping.json"
PSYCHOLOGY_IMAGES_MAPPING_PATH = PSYCHOLOGY_IMAGES_DIR / "psychology_images_mapping.json"
# 提取的图片文件路径
CAMPUS_IMAGES_PATH = CAMPUS_IMAGES_DIR / "campus_processed_images"
PSYCHOLOGY_IMAGES_PATH=PSYCHOLOGY_IMAGES_DIR /"psychology_processed_images"

# ==============================
# --- 目录验证 ---
# ==============================
if __name__ == '__main__':
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"知识库根目录: {KNOWLEDGE_BASE_DIR}")
    print(f"论文助手知识库: {PAPER_DOCS_DIR}")
    print(f"校园知识库: {CAMPUS_DOCS_DIR}")
    print(f"健身饮食知识库: {FITNESS_DOCS_DIR}")
    print(f"心理助手知识库: {PSYCHOLOGY_DOCS_DIR}")
    print(f"FAISS索引根目录: {FAISS_INDEX_DIR}")
    print(f"论文助手索引：{PAPER_INDEX_DIR}")
    print(f"校园知识索引：{CAMPUS_INDEX_DIR}")
    print(f"健身饮食索引：{FITNESS_INDEX_DIR}")
    print(f"心理助手索引：{PSYCHOLOGY_INDEX_DIR}")