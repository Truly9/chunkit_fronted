"""
update_faiss_with_images.py

功能：将图片描述添加到FAISS向量数据库，并创建（更新）图片映射文件
使用场景：在运行ImageExtractor生成图片描述后，将图片描述添加到FAISS索引
主要功能：
  1. 加载图片描述JSON文件
  2. 将图片描述添加到FAISS数据库
  3. 创建图片ID到图片路径的映射文件
  4. 支持多场景处理（campus/psychology）
"""
import json
import base64
import os
import sys
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

# 添加上一级目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 获取 chunkit_fronted 目录
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 现在可以正确导入 faiss_store_y
try:
    from Text_Processor.faiss_store_y import FAISSVectorStore
    print("成功导入 FAISSVectorStore")
except ImportError as e:
    print(f"导入错误: {e}")
    print("无法导入 FAISSVectorStore，请检查文件路径")
    sys.exit(1)

from Utils.Path import (
        PAPER_DOCS_DIR, CAMPUS_DOCS_DIR, FITNESS_DOCS_DIR, PSYCHOLOGY_DOCS_DIR,
        PAPER_INDEX_DIR, CAMPUS_INDEX_DIR, FITNESS_INDEX_DIR, PSYCHOLOGY_INDEX_DIR,
        ALL_PROCESSED_IMAGES_DIR, CAMPUS_IMAGES_DIR, PAPER_IMAGES_DIR, FITNESS_IMAGES_DIR, PSYCHOLOGY_IMAGES_DIR,
        CAMPUS_PROCESSED_EXTRACTED_IMAGES,PSYCHOLOGY_PROCESSED_EXTRACTED_IMAGES,
        CAMPUS_EXTRACTED_IMAGES_JSON,PSYCHOLOGY_EXTRACTED_IMAGES_JSON,
        CAMPUS_IMAGES_PATH,PSYCHOLOGY_IMAGES_PATH,
        CAMPUS_IMAGES_MAPPING_PATH, PSYCHOLOGY_IMAGES_MAPPING_PATH  # 确保导入映射路径
    )

class ImageFAISSUpdater:
    """将图片描述添加到FAISS数据库的类"""

    def __init__(self, faiss_store_path: str = "faiss_index1"):
        self.faiss_store_path = faiss_store_path
        current_dir = os.path.dirname(os.path.abspath(__file__))
      
        # 使用本地的Qwen3 embedding模型
        self.embedding_model = SentenceTransformer(
        os.path.join(os.getcwd(), "Qwen3-Embedding-0___6B"),
        tokenizer_kwargs={"padding_side": "left"},
        trust_remote_code=True
        )

        # 初始化或加载FAISS存储
        try:
            self.faiss_store = FAISSVectorStore(index_path=faiss_store_path)
            print(f"FAISS存储初始化完成: {faiss_store_path}")
        except Exception as e:
            print(f"初始化FAISS存储时出错: {e}")
            self.faiss_store = FAISSVectorStore(index_path=faiss_store_path)

    def load_processed_images(self, json_path: str) -> List[Dict]:
        """从JSON文件加载处理后的图片数据"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载JSON文件时出错: {e}")
            return []

    def remove_existing_image_chunks(self, scene_prefix: str = ""):
        """删除现有的图片描述chunks，可以按场景前缀删除"""
        print(f"正在删除现有的图片描述chunks...")
        if scene_prefix:
            # 删除特定场景的图片chunks
            self.faiss_store.remove_by_id_prefix(f"image_{scene_prefix}_")
            print(f"场景 '{scene_prefix}' 的现有图片描述chunks已删除")
        else:
            # 删除所有图片chunks
            self.faiss_store.remove_by_id_prefix("image_")
            print("所有现有图片描述chunks已删除")

    def create_image_chunks(self, processed_images: List[Dict], scene_name: str = "") -> tuple[List[Dict], Dict[str, Dict]]:
        """将每个图片描述作为独立chunk，使用image x:前缀标签"""
        chunks = []
        image_mapping = {}  # 存储id到图片信息的映射

        for img_data in processed_images:
            # 使用图片哈希生成ID
            image_hash = img_data.get('image_hash', '')
            if scene_name:
                image_id = f"image_{scene_name}_{image_hash}"
            else:
                image_id = f"image_{image_hash}"

            # 在描述前加上image x:标签
            chunk_content = f"{image_id}: {img_data['enhanced_description']}"

            # 存储图片映射信息
            image_mapping[image_id] = {
                'image_path': img_data.get('image_path', ''),
                'image_filename': img_data.get('image_filename', ''),
                'source_file': img_data['source_file'],
                'context_before': img_data['context_before'],
                'context_after': img_data['context_after'],
                'ai_description': img_data.get('original_description', ''),
                'enhanced_description': img_data['enhanced_description'],
                'image_size': img_data.get('image_size', 0),
                'scene': scene_name  # 添加场景信息
            }

            # 创建简化的chunk字典
            chunk = {
                'content': chunk_content,
                'metadata': {
                    'type': 'image_description',
                    'image_id': image_id,
                    'scene': scene_name
                },
                'chunk_id': image_id  # 使用与image_id相同的ID
            }

            chunks.append(chunk)

        return chunks, image_mapping

    def create_image_chunks_with_paths(self, processed_images: List[Dict], scene_name: str = "") -> tuple[List[Dict], Dict[str, Dict]]:
        """将每个图片描述作为独立chunk，包含完整的路径信息"""
        chunks = []
        image_mapping = {}  # 存储id到图片信息的映射

        for img_data in processed_images:
            # 使用图片哈希生成ID
            image_hash = img_data.get('image_hash', '')
            if scene_name:
                image_id = f"image_{scene_name}_{image_hash}"
            else:
                image_id = f"image_{image_hash}"

            # 在描述前加上image x:标签
            chunk_content = f"{image_id}: {img_data['enhanced_description']}"

            # 存储图片映射信息（包含完整路径）
            image_mapping[image_id] = {
                'image_path': img_data.get('image_path', ''),
                'image_filename': img_data.get('image_filename', ''),
                'processed_path': img_data.get('processed_path', ''),
                'source_file': img_data['source_file'],
                'context_before': img_data['context_before'],
                'context_after': img_data['context_after'],
                'ai_description': img_data.get('original_description', ''),
                'enhanced_description': img_data['enhanced_description'],
                'scene': scene_name  # 添加场景信息
            }

            # 创建简化的chunk字典
            chunk = {
                'content': chunk_content,
                'metadata': {
                    'type': 'image_description',
                    'image_id': image_id,
                    'scene': scene_name
                },
                'chunk_id': image_id  # 使用与image_id相同的ID
            }

            chunks.append(chunk)

        return chunks, image_mapping

    def add_image_chunks_to_faiss(self, image_chunks: List[Dict]):
        """将图片chunks添加到FAISS数据库"""
        print(f"开始将 {len(image_chunks)} 个图片描述添加到FAISS数据库...")

        added_count = 0
        # 批量处理以提高效率
        batch_size = 10
        for i in range(0, len(image_chunks), batch_size):
            batch = image_chunks[i:i + batch_size]

            try:
                # 批量生成embedding
                contents = [chunk['content'] for chunk in batch]
                embeddings = self.embedding_model.encode(contents)
                metadatas = [chunk['metadata'] for chunk in batch]
                # 批量添加到FAISS存储
                documents = [chunk['content'] for chunk in batch]
                ids = [chunk['chunk_id'] for chunk in batch]

                self.faiss_store.add(
                    documents=documents,
                    embeddings=embeddings.tolist(),
                    ids=ids,
                    metadatas=metadatas
                )

                added_count += len(batch)
                print(f"已添加图片描述批次 {i // batch_size + 1}: {len(batch)} 个描述")

            except Exception as e:
                print(f"添加图片描述批次 {i // batch_size + 1} 时出错: {e}")
                continue

        print("图片描述添加完成！")
        return added_count

    def save_image_mapping(self, image_mapping: Dict[str, Dict], mapping_path: str = "image_mapping.json"):
        """保存图片ID到路径的映射文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(image_mapping, f, ensure_ascii=False, indent=2)
            print(f"图片映射文件已保存到: {mapping_path}")
            
            # 验证保存的文件
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                print(f"映射文件验证成功，包含 {len(saved_data)} 个图片")
                
                # 显示前几个ID用于验证
                if saved_data:
                    print("映射文件前5个ID:")
                    for i, (img_id, img_info) in enumerate(list(saved_data.items())[:5]):
                        path = img_info.get('image_path', 'N/A')
                        print(f"  {i+1}. {img_id}: {path}")
        except Exception as e:
            print(f"保存图片映射文件时出错: {e}")

    def save_faiss_index(self):
        """保存FAISS索引"""
        try:
            self.faiss_store.save()
            print(f"FAISS索引已保存到: {self.faiss_store_path}")
        except Exception as e:
            print(f"保存FAISS索引时出错: {e}")

    def get_faiss_stats(self):
        """获取FAISS数据库统计信息"""
        try:
            total_docs = self.faiss_store.count()
            print(f"FAISS数据库统计:")
            print(f"  总文档数: {total_docs}")

            # 统计图片类型的文档
            image_count = 0
            campus_count = 0
            psychology_count = 0
            other_count = 0
        
            # 直接访问 metadata 字典来统计
            for doc_id, metadata in self.faiss_store.metadata.items():
                # 检查是否是图片描述
                if metadata.get('type') == 'image_description':
                    image_count += 1
                    scene = metadata.get('scene', '')
                    if scene == 'campus':
                        campus_count += 1
                    elif scene == 'psychology':
                        psychology_count += 1
                    else:
                        other_count += 1
        
            print(f"  图片描述文档总数: {image_count}")
            if campus_count > 0:
                print(f"  - Campus场景图片: {campus_count}")
            if psychology_count > 0:
                print(f"  - Psychology场景图片: {psychology_count}")
            if other_count > 0:
                print(f"  - 其他场景图片: {other_count}")
            print(f"  文本文档数: {total_docs - image_count}")

            return {
                'total_chunks': total_docs,
                'image_chunks': image_count,
                'campus_image_chunks': campus_count,
                'psychology_image_chunks': psychology_count,
                'other_image_chunks': other_count,
                'text_chunks': total_docs - image_count
            }

        except Exception as e:
            print(f"获取统计信息时出错: {e}")
            return None


def process_scene(scene_name: str, json_path: str, faiss_store_path: str, mapping_path: str):
    """处理单个场景"""
    print(f"\n{'='*60}")
    print(f"开始处理 {scene_name.upper()} 场景的图片数据")
    print(f"{'='*60}")
    
    # 检查JSON文件是否存在
    if not os.path.exists(json_path):
        print(f"错误: 找不到处理后的图片数据文件: {json_path}")
        print(f"请先运行 image_processor.py 来处理 {scene_name} 场景的图片")
        return False

    # 创建FAISS更新器
    updater = ImageFAISSUpdater(faiss_store_path)

    # 删除该场景现有的图片描述chunks
    updater.remove_existing_image_chunks(scene_name)

    # 加载处理后的图片数据
    processed_images = updater.load_processed_images(json_path)

    if not processed_images:
        print(f"{scene_name.upper()} 场景没有找到处理后的图片数据")
        return False

    print(f"加载了 {len(processed_images)} 个 {scene_name} 场景的图片描述")

    # 创建图片chunks（每个图片描述作为独立chunk）
    image_chunks, image_mapping = updater.create_image_chunks(processed_images, scene_name)

    # 保存图片映射文件
    updater.save_image_mapping(image_mapping, mapping_path)

    # 添加到FAISS数据库
    added_count = updater.add_image_chunks_to_faiss(image_chunks)

    # 保存索引
    updater.save_faiss_index()

    print(f"\n{scene_name.upper()} 场景处理完成！")
    print(f"成功添加了 {added_count} 个图片描述到FAISS数据库")
    
    return True


def main():
    """主函数 - 处理多个场景"""
    
    # 处理 campus 场景
    campus_success = process_scene(
        scene_name="campus",
        json_path=str(CAMPUS_EXTRACTED_IMAGES_JSON),
        faiss_store_path=str(CAMPUS_INDEX_DIR),
        mapping_path=str(CAMPUS_IMAGES_MAPPING_PATH)
    )
    
    # 处理 psychology 场景
    psychology_success = process_scene(
        scene_name="psychology", 
        json_path=str(PSYCHOLOGY_EXTRACTED_IMAGES_JSON),
        faiss_store_path=str(PSYCHOLOGY_INDEX_DIR),
        mapping_path=str(PSYCHOLOGY_IMAGES_MAPPING_PATH)
    )
    
    # 显示总体统计信息（使用最后一个处理的FAISS存储）
    print(f"\n{'='*60}")
    print("FAISS数据库总体统计")
    print(f"{'='*60}")
    
    if campus_success:
        campus_updater = ImageFAISSUpdater(str(CAMPUS_INDEX_DIR))
        campus_stats = campus_updater.get_faiss_stats()
    
    if psychology_success:
        psychology_updater = ImageFAISSUpdater(str(PSYCHOLOGY_INDEX_DIR))
        psychology_stats = psychology_updater.get_faiss_stats()
    
    print(f"\n所有场景的图片描述已成功添加到FAISS数据库！")


if __name__ == "__main__":
        main()