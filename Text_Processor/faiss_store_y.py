import faiss
import numpy as np
import json
import os
import time
import sys
from typing import List, Dict, Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Utils.Path import (
    PAPER_DOCS_DIR, CAMPUS_DOCS_DIR, FITNESS_DOCS_DIR, PSYCHOLOGY_DOCS_DIR,
    PAPER_INDEX_DIR, CAMPUS_INDEX_DIR, FITNESS_INDEX_DIR, PSYCHOLOGY_INDEX_DIR,
    FAISS_INDEX_DIR
)

class FAISSVectorStore:
    """FAISS向量存储类"""
    
    def __init__(self, index_path: str, collection_name: str = "document_embeddings",
                 dimension: int = 1024, reset: bool = False, embedding_model=None):
        """
        初始化FAISS向量存储
        
        Args:
            index_path: 索引文件存储路径
            collection_name: 集合名称，用于生成索引文件名
            dimension: 向量维度
            reset: 是否重置索引
            embedding_model: 嵌入模型，用于文本编码
        """
        self.index_path = index_path
        self.collection_name = collection_name
        self.dimension = dimension
        self.embedding_model = embedding_model  # 添加embedding_model属性

        # 创建索引目录
        try:
            os.makedirs(index_path, exist_ok=True)
            print(f"已创建/确认目录: {index_path}")
        except Exception as e:
            print(f"警告: 无法在 {index_path} 创建目录: {e}")
            # 使用备用目录
            import tempfile
            self.index_path = os.path.join(tempfile.gettempdir(), "faiss_index", collection_name)
            os.makedirs(self.index_path, exist_ok=True)
            print(f"使用备用目录: {self.index_path}")

        # 索引文件路径 - 使用collection_name作为文件名
        self.index_file = os.path.join(self.index_path, f"{collection_name}.index")
        self.metadata_file = os.path.join(self.index_path, f"{collection_name}_metadata.json")

        print(f"索引文件路径: {self.index_file}")
        print(f"元数据文件路径: {self.metadata_file}")

        # 详细检查文件存在性
        index_exists = os.path.exists(self.index_file) and os.path.getsize(self.index_file) > 0
        metadata_exists = os.path.exists(self.metadata_file) and os.path.getsize(self.metadata_file) > 0
    
        print(f"索引文件存在且非空: {index_exists}")
        print(f"元数据文件存在且非空: {metadata_exists}")
        
        # 修复：如果reset为True，强制创建新索引
        if reset:
            print("重置标志为True，创建新索引...")
            self._create_new_index()
        elif not index_exists:
            print("索引文件不存在，创建新索引...")
            self._create_new_index()
        else:
            print("加载现有FAISS索引...")
            self._load_existing_index()
    
    def _create_new_index(self):
        """创建新的FAISS索引"""
        try:
            # 使用L2距离的平面索引
            import faiss
            import numpy as np
            
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {}
            self.id_to_idx = {}
            self.idx_to_id = {}
            self.documents = {}  # 添加documents属性
            self.next_idx = 0
            print(f"创建新的FAISS索引，维度: {self.dimension}")
            
            # 立即保存新创建的索引
            self.save()
            
        except Exception as e:
            print(f"创建新索引失败: {e}")
            raise
    
    def _load_existing_index(self):
        """加载现有的FAISS索引"""
        try:
            import faiss
            import numpy as np
            
            # 加载索引
            print(f"正在加载索引文件: {self.index_file}")
            self.index = faiss.read_index(self.index_file)
            
            # 加载元数据
            if os.path.exists(self.metadata_file):
                print(f"正在加载元数据文件: {self.metadata_file}")
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = data.get('metadata', {})
                    self.id_to_idx = data.get('id_to_idx', {})
                    self.idx_to_id = {str(v): k for k, v in self.id_to_idx.items()}
                    self.next_idx = data.get('next_idx', 0)
                    
                    # 从metadata中重建documents
                    self.documents = {}
                    for doc_id, meta in self.metadata.items():
                        self.documents[doc_id] = meta.get('content', '')
            else:
                print("元数据文件不存在，初始化空元数据")
                self.metadata = {}
                self.id_to_idx = {}
                self.idx_to_id = {}
                self.documents = {}
                self.next_idx = 0
            
            print(f"加载现有FAISS索引，当前文档数量: {self.index.ntotal}")
            print(f"元数据中的文档数量: {len(self.metadata)}")
            print(f"映射表中的文档数量: {len(self.id_to_idx)}")
            print(f"文档内容数量: {len(self.documents)}")
            
            # 验证索引和元数据的一致性
            if self.index.ntotal != len(self.metadata):
                print(f"⚠️  警告: 索引中的向量数({self.index.ntotal})与元数据中的文档数({len(self.metadata)})不一致")
                
        except Exception as e:
            print(f"加载索引失败: {str(e)}")
            print("将创建新索引...")
            self._create_new_index()
    
    def add(self, documents: List[str], embeddings: List[List[float]], ids: List[str], 
            metadatas: Optional[List[Dict[str, Any]]] = None):
        """添加文档到索引"""
        if len(documents) != len(embeddings) or len(documents) != len(ids):
            raise ValueError("documents, embeddings, ids的长度必须相同")
    
        if metadatas and len(metadatas) != len(documents):
            raise ValueError("metadatas长度必须与documents相同")
    
        print(f"添加 {len(documents)} 个文档到索引")
        
        if len(documents) == 0:
            print("⚠️  没有文档可添加")
            return

        # 转换嵌入向量为numpy数组
        import numpy as np
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # 检查向量维度是否匹配
        if embeddings_array.shape[1] != self.dimension:
            print(f"❌ 向量维度不匹配: 期望{self.dimension}, 实际{embeddings_array.shape[1]}")
            return

        # 添加到FAISS索引
        start_idx = self.next_idx
        self.index.add(embeddings_array)
    
        # 更新映射和元数据
        for i, (doc_id, document) in enumerate(zip(ids, documents)):
            idx = start_idx + i
            self.id_to_idx[doc_id] = idx
            self.idx_to_id[str(idx)] = doc_id
            self.documents[doc_id] = document  # 添加到documents
        
            # 存储文档内容和元数据
            doc_metadata = {
                'content': document,
                'id': doc_id,
                'added_at': time.time()  # 添加时间戳
            }
            if metadatas and i < len(metadatas):
                doc_metadata.update(metadatas[i])
        
            self.metadata[doc_id] = doc_metadata
        
            # 调试信息
            if i < 3:  # 只打印前3个的调试信息
                print(f"  映射: idx={idx} -> id={doc_id}")
    
        self.next_idx += len(documents)
    
        # 立即保存以确保数据持久化
        self.save()
    
        print(f"添加完成: 总文档数={self.count()}, 下一个索引={self.next_idx}")
    
    def search(self, query, top_k=5):
        """搜索相似的文档
    
        Args:
            query: 查询文本或查询向量
            top_k: 返回最相似的k个结果
        """
        try:
            # 如果query是字符串，先编码为向量
            if isinstance(query, str):
                if self.embedding_model is None:
                    raise ValueError("查询是字符串但没有提供embedding_model")
                print(f"编码查询文本: '{query}'")
                query_embedding = self.embedding_model.encode([query])[0].tolist()
            else:
                query_embedding = query
            
            print(f"FAISS搜索: top_k={top_k}, 查询向量长度={len(query_embedding)}")
        
            query_vector = np.array([query_embedding]).astype('float32')
        
            # 执行搜索
            distances, indices = self.index.search(query_vector, top_k)
        
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1:
                    # 通过索引找到文档ID
                    str_idx = str(idx)
                    if str_idx in self.idx_to_id:
                        doc_id = self.idx_to_id[str_idx]
                        document = self.documents.get(doc_id, '')
                        metadata = self.metadata.get(doc_id, {})
                    
                        results.append({
                            'id': doc_id,
                            'document': document,
                            'metadata': metadata,
                            'score': float(1 - distance)  # 转换为相似度分数
                        })
        
            print(f"搜索完成，找到 {len(results)} 个结果")
            return results
        
        except Exception as e:
            print(f"❌ FAISS搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        return self.metadata.get(doc_id)
    
    def delete(self, doc_ids: List[str]):
        """删除文档（注意：FAISS不支持真正的删除，这里只是从元数据中移除）"""
        deleted_count = 0
        for doc_id in doc_ids:
            if doc_id in self.metadata:
                del self.metadata[doc_id]
                deleted_count += 1
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                del self.id_to_idx[doc_id]
                if str(idx) in self.idx_to_id:
                    del self.idx_to_id[str(idx)]
            if doc_id in self.documents:
                del self.documents[doc_id]
        
        if deleted_count > 0:
            print(f"删除了 {deleted_count} 个文档")
            # 保存更新后的元数据
            self.save()
        else:
            print("没有找到要删除的文档")
    
    def count(self) -> int:
        """返回文档数量"""
        return len(self.metadata)
    
    def save(self):
        """保存索引和元数据到磁盘，提供详细的错误处理"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        
            print(f"尝试保存FAISS索引到: {self.index_file}")
            print(f"尝试保存元数据到: {self.metadata_file}")
        
            # 检查目录权限
            index_dir = os.path.dirname(self.index_file)
            if not os.access(index_dir, os.W_OK):
                raise PermissionError(f"没有写入权限: {index_dir}")
        
            # 检查文件是否被锁定
            try:
                with open(self.index_file, 'a') as f:
                    pass
            except IOError as e:
                raise IOError(f"文件可能被锁定: {self.index_file}, 错误: {e}")
        
            # 尝试保存FAISS索引
            print("开始写入FAISS索引...")
            import faiss
            faiss.write_index(self.index, self.index_file)
            print("FAISS索引写入成功")
        
            # 保存元数据
            metadata_to_save = {
                'metadata': self.metadata,
                'id_to_idx': self.id_to_idx,
                'next_idx': self.next_idx,
                'saved_at': time.time(),
                'collection_name': self.collection_name,
                'dimension': self.dimension
            }
        
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)
            
            print("元数据保存成功!")
            print(f"保存完成: {len(self.metadata)} 个文档")
        
        except Exception as e:
            print(f"保存FAISS索引失败: {str(e)}")
            print("将使用内存模式，但数据不会持久化")
            # 不抛出异常，让程序继续运行
    
    def clear(self):
        """清空所有数据"""
        print("清空所有索引数据...")
        self._create_new_index()
        print("索引数据已清空")
    
    def remove_by_id_prefix(self, prefix: str):
        """根据ID前缀删除文档"""
        ids_to_remove = [doc_id for doc_id in self.metadata.keys() if doc_id.startswith(prefix)]
        if ids_to_remove:
            self.delete(ids_to_remove)
            print(f"删除了 {len(ids_to_remove)} 个以 '{prefix}' 开头的文档")
        else:
            print(f"没有找到以 '{prefix}' 开头的文档")
    
    @property
    def ids(self):
        """返回所有文档ID列表"""
        return list(self.metadata.keys())
    
    def get_stats(self):
        """获取索引统计信息"""
        import faiss
        stats = {
            'total_documents': self.count(),
            'total_vectors': self.index.ntotal if hasattr(self, 'index') and self.index else 0,
            'dimension': self.dimension,
            'collection_name': self.collection_name,
            'index_path': self.index_path,
            'next_index': self.next_idx
        }
        return stats
    
    def print_stats(self):
        """打印索引统计信息"""
        stats = self.get_stats()
        print("=== FAISS索引统计 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")