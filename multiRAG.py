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
import time
from transformers import AutoModel
from sentence_transformers import SentenceTransformer, CrossEncoder

# 导入现有模块
from Text_Processor.faiss_store_y import FAISSVectorStore
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
                 cross_encoder_path: str = "cross-encoder-model",
                 debug: bool = False):
        """
        初始化MultiRAG系统
    
        Args:
            scene: 场景类型 ("campus" 或 "psychology"或"fitness"或"paper")
            embedding_model_path: 嵌入模型路径
            cross_encoder_path: 交叉编码器路径
            debug: 是否启用调试模式
        """
        self.scene = scene
        self.embedding_model_path = embedding_model_path
        self.cross_encoder_path = cross_encoder_path
        self.debug = debug
        
        # 根据场景设置默认路径 - 统一路径格式
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
            base_dir = "D:\\New_project_10-master\\New_project_10-master\\demo\\back-end-python\\chunkit_fronted"
            self.index_path = os.path.join(base_dir, "faiss_index", scene)
            self.image_output_dir = os.path.join(base_dir, "All_Processed_Images", scene, f"{scene}_processed_images")
            self.image_mapping_file = os.path.join(base_dir, "All_Processed_Images", scene, f"{scene}_images_mapping.json")
            self.collection_name = f"{scene}_docs"
        
        # 初始化模型（延迟加载）
        self._embedding_model = None
        self._cross_encoder = None
        self._vector_store = None
        self._text_splitter = None
        
        # 初始化处理状态跟踪
        self._processed_files: Set[str] = set()
        self._processed_images: Set[str] = set()
        self._image_mapping: Dict = {}
        
        # 打印调试信息
        if self.debug:
            print(f"MultiRAG 初始化调试信息:")
            print(f"  Scene: {self.scene}")
            print(f"  Index Path: {self.index_path}")
            print(f"  Image Mapping File: {self.image_mapping_file}")
            print(f"  Image Mapping File Exists: {os.path.exists(self.image_mapping_file)}")
            
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
            print(f"创建新的图片映射文件: {self.image_mapping_file}")
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
                if self.debug:
                    print(f"加载处理记录: {len(self._processed_files)} 个文件, {len(self._processed_images)} 个图片")
            except Exception as e:
                print(f"加载处理记录失败: {e}")
                self._processed_files = set()
                self._processed_images = set()
        else:
            self._processed_files = set()
            self._processed_images = set()

    def _ensure_processed_files_loaded(self):
        """确保已处理的文件记录已加载"""
        if not self._processed_files or not self._processed_images:
            self._load_processed_files()

    def _save_processed_files(self):
        """保存已处理的文件记录 - 修复版本"""
        processed_files_file = os.path.join(self.index_path, f"processed_files_{self.scene}.json")
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(processed_files_file), exist_ok=True)
        
            data = {
                'files': list(self._processed_files),
                'images': list(self._processed_images),
                'last_updated': time.time(),
                'total_files': len(self._processed_files),
                'total_images': len(self._processed_images)
            }
        
            # 使用原子写入避免文件损坏
            temp_file = processed_files_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
            # 替换原文件
            if os.path.exists(processed_files_file):
                os.remove(processed_files_file)
            os.rename(temp_file, processed_files_file)
        
            print(f"处理记录已保存: {len(self._processed_files)} 个文件, {len(self._processed_images)} 个图片")
        
        except Exception as e:
            print(f"保存已处理文件记录失败: {e}")
            # 尝试直接写入作为后备
            try:
                with open(processed_files_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'files': list(self._processed_files),
                        'images': list(self._processed_images)
                    }, f, ensure_ascii=False, indent=2)
            except:
                pass

    def _get_file_hash(self, file_path: str) -> str:
        """计算文件哈希值用于去重"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return os.path.basename(file_path)  # 回退到文件名
        
    def _deduplicate_chunks(self, chunks):
        """对文本块进行去重"""
        seen = set()
        deduplicated = []
    
        for chunk in chunks:
            # 标准化文本（去除多余空格，统一大小写）
            normalized = ' '.join(chunk.strip().split()).lower()
            chunk_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        
            if chunk_hash not in seen:
                seen.add(chunk_hash)
                deduplicated.append(chunk)
            else:
                if self.debug:
                    print(f"去重重复文本块: {chunk[:100]}...")
    
        return deduplicated

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
                if self.debug:
                    print(f"嵌入模型加载成功: {self.embedding_model_path}")
            except Exception as e:
                print(f"加载嵌入模型失败: {e}")
                raise
        return self._embedding_model

    @property
    def vector_store(self):
        """向量存储属性（延迟加载）- 只使用场景特定索引"""
        if self._vector_store is None:
            try:
                # 只使用场景特定索引
                scene_index_file = os.path.join(self.index_path, f"{self.collection_name}.index")
                
                if os.path.exists(scene_index_file) and os.path.getsize(scene_index_file) > 0:
                    if self.debug:
                        print(f"使用场景特定索引: {scene_index_file}")
                    actual_collection_name = self.collection_name
                else:
                    # 如果场景特定索引不存在，创建新的场景特定索引
                    if self.debug:
                        print(f"场景特定索引不存在，创建新的: {scene_index_file}")
                    actual_collection_name = self.collection_name
                
                # 初始化向量存储
                self._vector_store = FAISSVectorStore(
                    index_path=self.index_path,
                    collection_name=actual_collection_name
                )
                
                doc_count = self._vector_store.count()
                if self.debug:
                    print(f"向量存储初始化完成，包含 {doc_count} 个文档")
                    
                # 检查文档类型分布
                image_count = 0
                text_count = 0
                for doc_id in self._vector_store.ids:
                    if doc_id.startswith(f'image_{self.scene}_'):
                        image_count += 1
                    else:
                        text_count += 1
                        
                print(f"文档类型分布: {image_count} 个图片, {text_count} 个文本")
                
            except Exception as e:
                print(f"加载向量存储失败: {e}")
                # 创建新的向量存储作为后备
                self._vector_store = FAISSVectorStore(
                    index_path=self.index_path,
                    collection_name=self.collection_name,
                    dimension=1024,
                    reset=True
                )
                print("创建了新的向量存储")
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
    
    def build(self, source: str, reset: bool = False):
        """
        对source文件夹中的所有文件建立数据库 - 修复增量版本
    
        Args:
            source: 源文件夹路径
            reset: 是否重置数据库（默认False，增量处理）
        """
        if not os.path.isdir(source):
            raise NotADirectoryError(f"文件夹 {source} 不存在或不是一个有效的文件夹")
    
        print(f"=== 开始{'重建' if reset else '增量构建'}{self.scene}场景多模态RAG数据库 ===")
    
        # 确保所有必要的目录和文件存在
        self._ensure_directories()
        self._initialize_files()
    
        # 检查索引状态
        status = self.check_index_status()
        print(f"当前索引状态: {status['documents_count']} 个文档")
    
        if reset:
            print("重置模式：清空现有索引...")
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
            self._save_processed_files()
        
            # 清空图片映射
            self._image_mapping = {}
            self._save_image_mapping()
        else:
            print(f"增量模式：保留现有 {status['documents_count']} 个文档")
            # 确保向量存储已加载（会加载现有索引）
            _ = self.vector_store
    
        # 1. 处理文本文档
        print("\n步骤1: 处理文本文档...")
        new_text_files = self._process_text_documents(source, incremental=not reset)
    
        # 2. 处理图片
        print("\n步骤2: 处理图片...")
        processed_images = self._process_images(source, incremental=not reset)
    
        # 3. 将图片描述添加到数据库
        if processed_images:
            print("\n步骤3: 将图片描述添加到数据库...")
            self._add_images_to_database(processed_images, incremental=not reset)
    
        # 保存处理记录
        self._save_processed_files()
    
        print(f"\n=== {self.scene}场景数据库{'重建' if reset else '增量构建'}完成 ===")
        self._print_database_stats()

    def insert(self, source: str):
        """
        将source文件夹中的所有文件加入知识库（增量添加）- 简化版本
        """
        print(f"=== 开始增量添加文档到{self.scene}场景数据库 ===")
        return self.build(source, reset=False)
    

    ###########################################################################
    # 文本处理相关方法
    ###########################################################################
    
    def _process_text_documents(self, source_folder: str, incremental: bool = False) -> int:
        """处理文本文档 - 修复增量版本"""
        # 确保已加载处理记录
        self._ensure_processed_files_loaded()

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
        skipped_files = 0
        processed_count = 0

        for file_idx, file_path in enumerate(all_files, 1):
            try:
                file_hash = self._get_file_hash(file_path)
        
                # 检查文件是否已处理（增量模式）
                if incremental:
                    if file_hash in self._processed_files:
                        if self.debug:
                            print(f"跳过已处理文件 ({file_idx}/{len(all_files)}): {os.path.basename(file_path)}")
                        skipped_files += 1
                        continue
        
                print(f"处理第 {file_idx}/{len(all_files)} 个文件: {os.path.basename(file_path)}")
        
                # 分割文档
                chunks = self._split_document(file_path)
                chunks = self._deduplicate_chunks(chunks)
                if not chunks:
                    print(f"  文件内容为空，跳过")
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
        
                # 标记为已处理
                self._processed_files.add(file_hash)
                new_files += 1
                print(f"  文件处理完成，新增 {len(chunks)} 个片段")
            
            except Exception as e:
                print(f"处理文件 {os.path.basename(file_path)} 时出错: {e}")
                continue

        # 在处理完成后一次性保存处理记录
        if new_files > 0:
            self._save_processed_files()

        print(f"文本处理完成: 新增 {new_files} 个文件，跳过 {skipped_files} 个已处理文件")
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
        """处理图片 - 修复增量版本"""
        # 使用现有的ImageExtractor
        extractor = ImageExtractor(source_folder, output_dir=self.image_output_dir)

        # 传递已处理的图片哈希集合给ImageExtractor
        processed_hashes = self._processed_images if incremental else set()
        processed_data = extractor.process_all_documents(processed_hashes=processed_hashes)

        if not processed_data:
            return []

        # 直接使用ImageExtractor处理的结果
        saved_images = []
        for img_data in processed_data:
            try:
                # 获取图片哈希
                image_hash = img_data.get('image_hash', 
                                    hashlib.md5(img_data.get('image_data', b'')).hexdigest()[:16])
            
                # 检查是否已经处理过（增量模式）- 双重检查
                if incremental and image_hash in self._processed_images:
                    if self.debug:
                        print(f"跳过已处理图片 (哈希: {image_hash})")
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
                # 添加到已处理集合
                self._processed_images.add(image_hash)
        
            except Exception as e:
                print(f"处理图片数据时出错: {e}")
                continue

        print(f"图片处理完成: 新增 {len(saved_images)} 个图片")
        return saved_images
    
    def _load_image_mapping(self) -> Dict:
        """加载图片映射文件"""
        try:
            if os.path.exists(self.image_mapping_file):
                with open(self.image_mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                    print(f"成功加载图片映射文件，包含 {len(mapping)} 个图片条目")
                    
                    # 验证映射文件内容
                    valid_count = 0
                    for img_id, img_info in mapping.items():
                        if isinstance(img_info, dict) and img_info.get('image_path'):
                            valid_count += 1
                    
                    print(f"有效图片条目: {valid_count}/{len(mapping)}")
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
            if self.debug:
                print(f"图片映射文件已保存: {self.image_mapping_file}")
        except Exception as e:
            print(f"保存图片映射文件失败: {e}")

    def _add_images_to_database(self, processed_images: List[Dict], incremental: bool = False):
        """将图片描述添加到数据库 - 修复增量版本"""
        if not processed_images:
            return

        # 确保图片映射已加载
        if not self._image_mapping:
            self._load_image_mapping()

        # 创建图片chunks和映射
        image_chunks = []
        new_image_mapping = {}

        for img_data in processed_images:
            try:
                image_hash = img_data.get('image_hash', '')
                image_id = self._generate_image_id(image_hash)
        
                # 检查是否已存在（增量模式）
                if incremental and image_id in self._image_mapping:
                    if self.debug:
                        print(f"跳过已存在的图片: {image_id}")
                    continue
        
                # 在描述前加上image x:标签
                chunk_content = f"{image_id}: {img_data['enhanced_description']}"
        
                # 确保图片路径是绝对路径
                image_path = img_data.get('image_path', '')
                if image_path and not os.path.isabs(image_path):
                    image_path = os.path.abspath(image_path)
        
                # 存储图片映射信息
                new_image_mapping[image_id] = {
                    'image_path': image_path,
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
                if self.debug:
                    print(f"准备添加图片: {image_id}")
        
            except Exception as e:
                print(f"处理图片数据时出错: {e}")
                continue

        # 更新图片映射文件
        if incremental:
            # 增量模式：只添加新的图片映射
            self._image_mapping.update(new_image_mapping)
            print(f"增量更新图片映射: 添加 {len(new_image_mapping)} 个新图片")
        else:
            # 重建模式：完全替换
            self._image_mapping = new_image_mapping
            print(f"重建图片映射: 共 {len(new_image_mapping)} 个图片")

        self._save_image_mapping()

        # 添加到向量存储
        success_count = 0
        for chunk in image_chunks:
            try:
                # 检查是否已存在（增量模式）
                if incremental and self.vector_store.exists(chunk['chunk_id']):
                    if self.debug:
                        print(f"跳过已存在的图片chunk: {chunk['chunk_id']}")
                    continue
            
                # 生成embedding
                embedding = self.embedding_model.encode([chunk['content']])[0]
        
                # 添加到FAISS存储
                self.vector_store.add(
                    documents=[chunk['content']],
                    embeddings=[embedding.tolist()],
                    ids=[chunk['chunk_id']]
                )
                success_count += 1
                if self.debug:
                    print(f"成功添加图片chunk: {chunk['chunk_id']}")
            except Exception as e:
                print(f"添加图片chunk到向量存储失败: {e}")
                continue

        # 在处理完成后保存处理记录
        if success_count > 0:
            self._save_processed_files()

        print(f"图片处理完成: 成功添加 {success_count}/{len(image_chunks)} 个图片描述")

    def _generate_image_id(self, image_hash: str) -> str:
        """生成图片ID"""
        return f"image_{self.scene}_{image_hash}"

    ###########################################################################
    # 检索方法
    ###########################################################################
    
    def retrieve(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        """检索-同时搜索文本和图片索引"""
        try:
            print(f"\n=== 开始检索: '{query}' ===")
        
            # 1. 搜索文本索引
            print("搜索文本索引...")
            text_results = self._search_text_index(query, topk)
            
            # 2. 搜索图片索引  
            print("搜索图片索引...")
            image_results = self._search_image_index(query, max(1, topk//2))
            
            # 3. 合并结果
            all_results = text_results + image_results
            all_results.sort(key=lambda x: x['score'], reverse=True)
            
            # 4. 返回topk个结果
            final_results = all_results[:topk]
            
            # 统计
            image_count = len([r for r in final_results if r['type'] == 1])
            text_count = len([r for r in final_results if r['type'] == 0])
            print(f"\n✅ 检索完成: {image_count} 个图片, {text_count} 个文本")
            
            return final_results
            
        except Exception as e:
            print(f"❌ 检索过程出错: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _search_text_index(self, query: str, topk: int) -> List[Dict]:
        """搜索文本索引"""
        try:
            # 使用现有的向量存储，不要重新初始化
            query_embedding = self.embedding_model.encode(
                query, prompt_name="query", convert_to_tensor=False, normalize_embeddings=True
            ).tolist()
            
            results = self.vector_store.search(query_embedding, topk)
            
            formatted_results = []
            for result in results:
                # 只处理文本类型的文档
                doc_id = result.get('id', '')
                if not doc_id.startswith('image_'):
                    formatted_results.append({
                        "type": 0,  # 文本类型
                        "document": result.get('content', ''),
                        "source": "",
                        "score": result.get('score', 0),
                        "content": result.get('content', ''),
                        "id": doc_id
                    })
                
            print(f"文本索引搜索完成: {len(formatted_results)} 个结果")
            return formatted_results
            
        except Exception as e:
            print(f"文本索引搜索失败: {e}")
            return []

    def _search_image_index(self, query: str, topk: int) -> List[Dict]:
        """搜索图片索引 - 修复图片路径版本"""
        try:
            print(f"开始图片索引搜索: query='{query}', topk={topk}")
            
            # 使用现有的向量存储
            query_embedding = self.embedding_model.encode(
                query, prompt_name="query", convert_to_tensor=False, normalize_embeddings=True
            ).tolist()
            
            # 搜索更多结果以确保找到图片
            search_topk = min(topk * 5, 100)
            print(f"实际搜索数量: {search_topk}")
            
            results = self.vector_store.search(query_embedding, search_topk)
            print(f"FAISS搜索返回 {len(results)} 个原始结果")
            
            # 加载图片映射
            image_mapping = self._load_image_mapping()
            print(f"加载的图片映射包含 {len(image_mapping)} 个条目")
            
            formatted_results = []
            image_count = 0
            processed_ids = set()
            
            for result in results:
                doc_id = result.get('id', '')
                content = result.get('content', '')
                score = result.get('score', 0)
                
                # 只处理图片类型的文档
                if doc_id.startswith('image_') and doc_id not in processed_ids:
                    image_count += 1
                    processed_ids.add(doc_id)
                    
                    # 从映射文件中获取图片信息
                    img_info = image_mapping.get(doc_id, {})
                    img_path = img_info.get('image_path', '')
                    description = img_info.get('enhanced_description', content)
                    
                    # 确保返回绝对路径
                    if img_path and not os.path.isabs(img_path):
                        img_path = os.path.abspath(img_path)
                    
                    # 检查图片文件是否存在
                    file_exists = img_path and os.path.exists(img_path)
                    
                    # 构建详细的图片信息，包含真实路径
                    result_data = {
                        "type": 1,  # 图片类型
                        "document": description,
                        "source": img_path if file_exists else "",
                        "score": score,
                        "content": content,
                        "id": doc_id,
                        "filename": os.path.basename(img_path) if img_path else "",
                        "full_path": img_path if file_exists else "",
                        "exists": file_exists
                    }
                    
                    if file_exists:
                        formatted_results.append(result_data)
                        print(f"✅ 找到有效图片: {os.path.basename(img_path)} (分数: {score:.4f})")
                    else:
                        if img_path:
                            print(f"❌ 图片文件不存在: {img_path}")
                        else:
                            print(f"❌ 图片无路径信息: {doc_id}")
                    
                    # 如果已经找到足够的图片，提前停止
                    if len(formatted_results) >= topk:
                        break
            
            print(f"图片索引搜索完成: 找到 {len(formatted_results)} 个有效图片 (从 {image_count} 个图片文档中)")
            return formatted_results
            
        except Exception as e:
            print(f"❌ 图片索引搜索失败: {e}")
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

    def _check_index_status(self):
        """检查索引状态"""
        try:
            # 只检查当前使用的索引
            text_index_path = os.path.join(self.index_path, f"{self.collection_name}.index")
            text_metadata_path = os.path.join(self.index_path, f"{self.collection_name}_metadata.json")
            
            text_index_exists = os.path.exists(text_index_path) and os.path.getsize(text_index_path) > 0
            text_metadata_exists = os.path.exists(text_metadata_path) and os.path.getsize(text_metadata_path) > 0
            
            if text_index_exists and text_metadata_exists:
                self._load_text_index()
                doc_count = len(self.text_metadata) if hasattr(self, 'text_metadata') else 0
                print(f"索引状态: {doc_count} 个文档")
            else:
                print("索引状态: 未初始化")
                
        except Exception as e:
            print(f"检查索引状态时出错: {e}")
            print("索引状态: 检查失败")

    def print_index_status(self):
        """打印索引状态"""
        status = self.check_index_status()
        print(f"\n=== {self.scene}场景索引状态 ===")
        print(f"索引路径: {status['index_path']}")
        print(f"索引存在: {status['index_exists']}")
        print(f"文档数量: {status['documents_count']}")
    
        if status['index_files']:
            print("索引文件:")
            for file_info in status['index_files']:
                print(f"  - {file_info['name']}: {file_info['size_human']}")
        else:
            print("警告: 未找到索引文件!")

# 使用示例
if __name__ == "__main__":
    # 1. 创建 MultiRAG 实例
    campus_rag = MultiRAG(scene="campus", debug=True)
    psychology_rag = MultiRAG(scene="psychology", debug=True)

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