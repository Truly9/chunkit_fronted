#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态RAG系统封装类 
提供build、insert、retrieve三个核心功能接口
支持campus和psychology双场景
"""

import os
import json
import hashlib
import PyPDF2
from docx import Document
from typing import List, Dict, Any, Set
from pathlib import Path
import torch
from transformers import AutoModel
from sentence_transformers import SentenceTransformer, CrossEncoder

# 导入现有模块
from faiss_store_y import FAISSVectorStore
from Text_Processor.textsplitters import RecursiveCharacterTextSplitter
from Image_Processor.Image_Process import ImageExtractor


from Utils.Path import (
    CAMPUS_DOCS_DIR, PSYCHOLOGY_DOCS_DIR,
    CAMPUS_INDEX_DIR, PSYCHOLOGY_INDEX_DIR,
    CAMPUS_IMAGES_PATH, PSYCHOLOGY_IMAGES_PATH,
    CAMPUS_IMAGES_MAPPING_PATH, PSYCHOLOGY_IMAGES_MAPPING_PATH,
    CAMPUS_EXTRACTED_IMAGES_JSON, PSYCHOLOGY_EXTRACTED_IMAGES_JSON
)

class MultiRAG:
    """
    多模态RAG系统封装类 
    
    核心功能:
    - build: 对source文件夹中的所有文件建立数据库
    - insert: 增量添加文件到知识库
    - retrieve: 检索相关度最高的topk个片段
    """
    
    def __init__(self, 
                 scene: str = "campus",
                 embedding_model_path: str = "Qwen3-Embedding-0___6B",
                 cross_encoder_path: str = "cross-encoder-model"):
        """
        初始化MultiRAG系统
    
        Args:
            scene: 场景类型 ("campus" 或 "psychology"或"fitness"或"paper")
            embedding_model_path: 嵌入模型路径
            cross_encoder_path: 交叉编码器路径
        """
        # 根据场景设置默认路径
        if scene == "campus":
            self.index_path = str(CAMPUS_INDEX_DIR)
            self.image_output_dir = str(CAMPUS_IMAGES_PATH)
            self.image_mapping_file = str(CAMPUS_IMAGES_MAPPING_PATH)
            self.collection_name = "campus_docs"
        elif scene == "psychology":
            self.index_path = str(PSYCHOLOGY_INDEX_DIR)
            self.image_output_dir = str(PSYCHOLOGY_IMAGES_PATH)
            self.image_mapping_file = str(PSYCHOLOGY_IMAGES_MAPPING_PATH)
            self.collection_name = "psychology_docs"
        else:
            self.index_path = f"str({scene}_INDEX_DIR)"
            self.image_output_dir = f"str({scene}_IMAGES_PATH)"
            self.image_mapping_file = f"str({scene}_IMAGES_MAPPING_PATH)"
            self.collection_name = f"{scene}_docs"
        
        self.scene = scene
        self.embedding_model_path = embedding_model_path
        self.cross_encoder_path = cross_encoder_path
        
        # 初始化模型（延迟加载）
        self._embedding_model = None
        self._cross_encoder = None
        self._vector_store = None
        self._text_splitter = None
        
        # 初始化处理状态跟踪
        self._processed_files: Set[str] = set()
        self._processed_images: Set[str] = set()
        
        # 确保所有必要的目录存在
        self._ensure_directories()
        
        # 初始化必要的文件
        self._initialize_files()
            
        print(f"MultiRAG系统初始化完成 - {self.scene}场景")
        print(f"索引路径: {self.index_path}")
        print(f"图片输出目录: {self.image_output_dir}")

    ###########################################################################
    # 初始化相关方法
    ###########################################################################
    
    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        directories = [
            self.index_path,
            self.image_output_dir,
            os.path.dirname(self.image_mapping_file) if os.path.dirname(self.image_mapping_file) else None
        ]
        
        for directory in directories:
            if directory and directory != ".":
                os.makedirs(directory, exist_ok=True)
    
    def _initialize_files(self):
        """初始化必要的文件"""
        # 初始化图片映射文件
        if not os.path.exists(self.image_mapping_file):
            with open(self.image_mapping_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        
        # 加载现有的图片映射
        self._load_image_mapping()
        
        # 加载已处理的文件记录
        self._load_processed_files()

    def _load_processed_files(self):
        """加载已处理的文件记录"""
        processed_files_file = os.path.join(self.index_path, f"processed_files_{self.scene}.json")
        if os.path.exists(processed_files_file):
            try:
                with open(processed_files_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._processed_files = set(data.get('files', []))
                    self._processed_images = set(data.get('images', []))
            except:
                self._processed_files = set()
                self._processed_images = set()

    def _save_processed_files(self):
        """保存已处理的文件记录"""
        processed_files_file = os.path.join(self.index_path, f"processed_files_{self.scene}.json")
        try:
            data = {
                'files': list(self._processed_files),
                'images': list(self._processed_images)
            }
            with open(processed_files_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存已处理文件记录失败: {e}")

    def _get_file_hash(self, file_path: str) -> str:
        """计算文件哈希值用于去重"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return os.path.basename(file_path)  # 回退到文件名

    ###########################################################################
    # 模型加载属性（延迟加载）
    ###########################################################################
    
    @property
    def embedding_model(self):
        """嵌入模型属性（延迟加载）"""
        if self._embedding_model is None:
            try:
                self._embedding_model = SentenceTransformer(
                    self.embedding_model_path,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"加载嵌入模型失败: {e}")
                raise
        return self._embedding_model

    @property
    def vector_store(self):
        """向量存储属性（延迟加载）"""
        if self._vector_store is None:
            try:
                self._vector_store = FAISSVectorStore(index_path=self.index_path)
            except Exception as e:
                print(f"加载向量存储失败: {e}")
                raise
        return self._vector_store

    @property
    def text_splitter(self):
        """文本分割器属性（延迟加载）"""
        if self._text_splitter is None:
            try:
                self._text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len,
                )
            except Exception as e:
                print(f"初始化文本分割器失败: {e}")
                raise
        return self._text_splitter
    
    ###########################################################################
    # 核心构建方法
    ###########################################################################
    
    def build(self, source: str):
        """
        对source文件夹中的所有文件建立数据库
        """
        if not os.path.isdir(source):
            raise NotADirectoryError(f"文件夹 {source} 不存在或不是一个有效的文件夹")
        
        print(f"=== 开始构建{self.scene}场景多模态RAG数据库 ===")
        
        # 确保所有必要的目录和文件存在
        self._ensure_directories()
        self._initialize_files()
        
        # 重置向量存储
        self._vector_store = FAISSVectorStore(
            index_path=self.index_path,
            collection_name=self.collection_name,
            dimension=1024,
            reset=True
        )
        
        # 清空处理记录
        self._processed_files.clear()
        self._processed_images.clear()
        
        # 1. 处理文本文档
        print("\n步骤1: 处理文本文档...")
        self._process_text_documents(source)
        
        # 2. 处理图片
        print("\n步骤2: 处理图片...")
        processed_images = self._process_images(source)
        
        # 3. 将图片描述添加到数据库
        if processed_images:
            print("\n步骤3: 将图片描述添加到数据库...")
            self._add_images_to_database(processed_images)
        
        # 保存处理记录
        self._save_processed_files()
        
        print(f"\n=== {self.scene}场景数据库构建完成 ===")
        self._print_database_stats()
    
    def insert(self, source: str):
        """
        将source文件夹中的所有文件加入知识库（增量添加）
        """
        if not os.path.isdir(source):
            raise NotADirectoryError(f"文件夹 {source} 不存在或不是一个有效的文件夹")
        
        print(f"=== 开始增量添加文档到{self.scene}场景数据库 ===")
        
        # 确保所有必要的目录和文件存在
        self._ensure_directories()
        self._initialize_files()
        
        # 1. 处理文本文档（增量添加）
        print("\n步骤1: 增量添加文本文档...")
        new_text_files = self._process_text_documents(source, incremental=True)
        
        # 2. 处理图片（增量添加）
        print("\n步骤2: 处理新图片...")
        processed_images = self._process_images(source, incremental=True)
        
        # 3. 将图片描述添加到数据库
        if processed_images:
            print("\n步骤3: 将新图片描述添加到数据库...")
            self._add_images_to_database(processed_images, incremental=True)
        
        # 4. 保存处理记录
        self._save_processed_files()
        
        print(f"\n=== {self.scene}场景文档增量添加完成 ===")
        print(f"新增文本文件: {new_text_files} 个")
        print(f"新增图片: {len(processed_images)} 个")
        self._print_database_stats()

    ###########################################################################
    # 文本处理相关方法
    ###########################################################################
    
    def _process_text_documents(self, source_folder: str, incremental: bool = False) -> int:
        """处理文本文档"""
        # 获取所有支持的文件
        supported_extensions = ['.txt', '.md', '.markdown', '.pdf', '.docx']
        all_files = []
    
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
            
                # 跳过临时文件和系统文件
                if self._should_skip_file(file):
                    continue
            
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    all_files.append(file_path)
    
        if not all_files:
            print("没有找到支持的文本文件")
            return 0
        
        print(f"找到 {len(all_files)} 个文本文件")
        
        new_files = 0
    
        for file_idx, file_path in enumerate(all_files, 1):
            try:
                # 检查是否已经处理过（增量模式）
                file_hash = self._get_file_hash(file_path)
                if incremental and file_hash in self._processed_files:
                    continue
            
                print(f"处理第 {file_idx}/{len(all_files)} 个文件: {os.path.basename(file_path)}")
            
                # 分割文档
                chunks = self._split_document(file_path)
                if not chunks:
                    continue
            
                # 生成嵌入向量
                embeddings = self.embedding_model.encode(chunks)
            
                # 准备数据
                documents = []
                embeddings_list = []
                ids = []
            
                for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    unique_id = self._generate_text_chunk_id(file_hash, chunk_idx)
                    documents.append(chunk)
                    embeddings_list.append(embedding.tolist())
                    ids.append(unique_id)
            
                # 添加到向量存储
                self.vector_store.add(
                    documents=documents,
                    embeddings=embeddings_list,
                    ids=ids
                )
            
                # 记录已处理文件
                self._processed_files.add(file_hash)
                new_files += 1
            
            except Exception as e:
                print(f"处理文件 {os.path.basename(file_path)} 时出错: {e}")
                continue
    
        return new_files

    def _split_document(self, filename: str) -> List[str]:
        """分割文档为chunks"""
        content = self._read_file(filename)
        if not content.strip():
            return []
        
        return self.text_splitter.split_text(content)

    def _read_file(self, filename: str) -> str:
        """读取文件内容，支持多种格式"""
        if filename.endswith('.txt') or filename.endswith('.md') or filename.endswith('.markdown'):
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    return file.read()
            except UnicodeDecodeError:
                try:
                    with open(filename, 'r', encoding='gbk') as file:
                        return file.read()
                except:
                    return ""
        
        elif filename.endswith('.pdf'):
            try:
                text = ""
                with open(filename, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() or ""
                return text
            except:
                return ""
        
        elif filename.endswith('.docx'):
            try:
                doc = Document(filename)
                return "\n".join([para.text for para in doc.paragraphs])
            except:
                return ""
        
        else:
            return ""

    def _should_skip_file(self, filename: str) -> bool:
        """判断是否应该跳过文件"""
        # 跳过Word临时文件
        if filename.startswith('~$'):
            return True
    
        # 跳过系统隐藏文件
        if filename.startswith('.'):
            return True
    
        # 跳过系统文件
        system_files = ['Thumbs.db', '.DS_Store']
        if filename in system_files:
            return True
    
        return False

    def _generate_text_chunk_id(self, file_hash: str, chunk_index: int) -> str:
        """生成文本chunk ID"""
        return f"text_{self.scene}_{file_hash}_chunk_{chunk_index}"

    ###########################################################################
    # 图片处理相关方法
    ###########################################################################
    
    def _process_images(self, source_folder: str, incremental: bool = False) -> List[Dict]:
        """处理图片"""
        # 使用现有的ImageExtractor
        extractor = ImageExtractor(source_folder, output_dir=self.image_output_dir)
        processed_data = extractor.process_all_documents()

        if not processed_data:
            return []

        # 直接使用ImageExtractor处理的结果
        saved_images = []
        for img_data in processed_data:
            try:
                # 获取图片哈希
                image_hash = img_data.get('image_hash', 
                                    hashlib.md5(img_data.get('image_data', b'')).hexdigest()[:16])
            
                # 检查是否已经处理过（增量模式）
                if incremental and image_hash in self._processed_images:
                    continue
            
                # 更新图片数据
                image_path = img_data.get('image_path', '')
                image_filename = img_data.get('image_filename', '')
            
                if not image_path and image_filename:
                    image_path = os.path.join(self.image_output_dir, image_filename)
            
                updated_img_data = img_data.copy()
                updated_img_data.update({
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'image_hash': image_hash
                })
            
                saved_images.append(updated_img_data)
                self._processed_images.add(image_hash)
            
            except:
                continue

        return saved_images

    def _load_image_mapping(self) -> Dict:
        """加载图片映射文件"""
        try:
            if os.path.exists(self.image_mapping_file):
                with open(self.image_mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                    print(f"成功加载图片映射文件，包含 {len(mapping)} 个图片条目")
                    return mapping
            else:
                print(f"图片映射文件不存在: {self.image_mapping_file}")
                return {}
        except Exception as e:
            print(f"加载图片映射文件失败: {e}")
            return {}
        
    def _save_image_mapping(self):
        """保存图片映射文件"""
        try:
            with open(self.image_mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self._image_mapping, f, ensure_ascii=False, indent=2)
        except:
            pass

    def _add_images_to_database(self, processed_images: List[Dict], incremental: bool = False):
        """将图片描述添加到数据库"""
        if not processed_images:
            return
    
        # 创建图片chunks和映射
        image_chunks = []
        image_mapping = {}
    
        for img_data in processed_images:
            # 使用统一的图片ID生成方法
            image_hash = img_data.get('image_hash', '')
            image_id = self._generate_image_id(image_hash)
        
            # 在描述前加上image x:标签
            chunk_content = f"{image_id}: {img_data['enhanced_description']}"
        
            # 存储图片映射信息
            image_mapping[image_id] = {
                'image_path': img_data.get('image_path', ''),
                'image_filename': img_data.get('image_filename', ''),
                'source_file': img_data['source_file'],
                'enhanced_description': img_data['enhanced_description'],
                'image_hash': image_hash
            }
        
            # 创建chunk
            chunk = {
                'content': chunk_content,
                'chunk_id': image_id
            }
        
            image_chunks.append(chunk)
    
        # 更新图片映射文件
        if incremental:
            self._image_mapping.update(image_mapping)
        else:
            self._image_mapping = image_mapping
        
        self._save_image_mapping()
    
        # 添加到向量存储
        for chunk in image_chunks:
            try:
                # 生成embedding
                embedding = self.embedding_model.encode([chunk['content']])[0]
                
                # 添加到FAISS存储
                self.vector_store.add(
                    documents=[chunk['content']],
                    embeddings=[embedding.tolist()],
                    ids=[chunk['chunk_id']]
                )
            except:
                continue

    def _generate_image_id(self, image_hash: str) -> str:
        """生成图片ID"""
        return f"image_{self.scene}_{image_hash}"

    ###########################################################################
    # 检索方法
    ###########################################################################
    
    def retrieve(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        """
        检索-使用图片映射文件
        """
        try:
            # 1. 生成查询向量
            query_embedding = self.embedding_model.encode(
                query,
                prompt_name="query", 
                convert_to_tensor=False,
                normalize_embeddings=True
            ).tolist()

            # 2. 从FAISS检索
            results = self.vector_store.search(query_embedding, topk)
            
            if not results:
                return []

            # 3. 加载图片映射文件
            image_mapping = self._load_image_mapping()
            
            # 4. 格式化输出并关联图片信息
            formatted_results = []
            for result in results:
                content = result.get('content', '')
                score = result.get('score', 0)
                
                # 检查是否是图片描述
                # 图片描述通常以 "image_" 开头，后面跟着场景和哈希值
                if content.startswith('image_'):
                    # 提取图片ID（格式：image_psychology_xxxx）
                    image_id = content.split(':', 1)[0].strip() if ':' in content else content.strip()
                    
                    print(f"发现图片内容: {image_id}")
                    
                    # 从映射文件中获取图片信息
                    img_info = image_mapping.get(image_id, {})
                    img_path = img_info.get('image_path', '')
                    description = img_info.get('enhanced_description', img_info.get('ai_description', ''))
                    
                    # 验证图片文件是否存在
                    if img_path and os.path.exists(img_path):
                        print(f"图片文件存在: {img_path}")
                        formatted_results.append({
                            "type": 1,  # 图片类型
                            "document": description,
                            "source": img_path,
                            "score": score,
                            "content": content
                        })
                    else:
                        print(f"图片文件不存在: {img_path}")
                        # 即使文件不存在，也返回图片描述
                        formatted_results.append({
                            "type": 1,
                            "document": content,
                            "source": "",
                            "score": score,
                            "content": content
                        })
                else:
                    # 纯文本
                    formatted_results.append({
                        "type": 0,  # 文本类型
                        "document": content,
                        "source": "",
                        "score": score,
                        "content": content
                    })

            # 打印调试信息
            image_count = len([r for r in formatted_results if r['type'] == 1])
            text_count = len([r for r in formatted_results if r['type'] == 0])
            print(f"检索结果统计: {image_count} 个图片, {text_count} 个文本")
            
            return formatted_results
            
        except Exception as e:
            print(f"检索过程出错: {e}")
            import traceback
            traceback.print_exc()
            return []

    
    ###########################################################################
    # 辅助方法
    ###########################################################################
    
    def _print_database_stats(self):
        """打印数据库统计信息"""
        total_docs = self.vector_store.count()
    
        # 统计图片和文本文档数量
        image_count = 0
        text_count = 0
    
        for doc_id in self.vector_store.ids:
            if doc_id.startswith(f'image_{self.scene}_'):
                image_count += 1
            else:
                text_count += 1
    
        print(f"\n{self.scene}场景数据库统计信息:")
        print(f"  总文档数: {total_docs}")
        print(f"  文本片段数: {text_count}")
        print(f"  图片描述数: {image_count}")
        print(f"  已处理文件: {len(self._processed_files)}")
        print(f"  已处理图片: {len(self._processed_images)}")


# 使用示例
if __name__ == "__main__":
    # 1. 创建 MultiRAG 实例
    campus_rag = MultiRAG(scene="campus")
    psychology_rag = MultiRAG(scene="psychology")

    # 2. 构建数据库
    campus_rag.build(str(CAMPUS_DOCS_DIR))
    psychology_rag.build(str(PSYCHOLOGY_DOCS_DIR))
    
    # 3. 测试检索功能
    campus_results = campus_rag.retrieve("校园邮箱如何使用", 5)
    print(f"校园场景检索结果: {len(campus_results)} 个")
    
    psychology_results = psychology_rag.retrieve("如何缓解焦虑", 5)
    print(f"心理学场景检索结果: {len(psychology_results)} 个")
    
    # 4. 增量添加文档
    campus_rag.insert(str(CAMPUS_DOCS_DIR))
    psychology_rag.insert(str(PSYCHOLOGY_DOCS_DIR))