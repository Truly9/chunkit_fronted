#--重构为基类+子类架构，消除代码重复，统一所有助手的图片检索、流式输出和错误处理逻辑--#
from dashscope import Application
from http import HTTPStatus
import os
import sys
from abc import ABC, abstractmethod

multiRAG_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, multiRAG_dir)
from multiRAG import MultiRAG
from ClassAssistant.LLMmodel import LLM_compus, LLM_psychology, LLM_paper, LLM_fitness


from Utils.Path import (
    PAPER_DOCS_DIR, CAMPUS_DOCS_DIR, FITNESS_DOCS_DIR, PSYCHOLOGY_DOCS_DIR,
    PAPER_INDEX_DIR, CAMPUS_INDEX_DIR, FITNESS_INDEX_DIR, PSYCHOLOGY_INDEX_DIR,
    ALL_PROCESSED_IMAGES_DIR, CAMPUS_IMAGES_DIR, PAPER_IMAGES_DIR, FITNESS_IMAGES_DIR, PSYCHOLOGY_IMAGES_DIR,
    CAMPUS_PROCESSED_EXTRACTED_IMAGES, PSYCHOLOGY_PROCESSED_EXTRACTED_IMAGES,
    CAMPUS_EXTRACTED_IMAGES_JSON, PSYCHOLOGY_EXTRACTED_IMAGES_JSON,
    CAMPUS_IMAGES_PATH, PSYCHOLOGY_IMAGES_PATH,
    CAMPUS_IMAGES_MAPPING_PATH, PSYCHOLOGY_IMAGES_MAPPING_PATH
)

APP_ID = "c2affdebf6664d438a4043216ee15dea"
apiKey = "sk-f89e754d6cff4f31a25f609e82b3bce1"

#--重构为基类+子类架构，消除代码重复，统一所有助手的图片检索、流式输出和错误处理逻辑--#
from dashscope import Application
from http import HTTPStatus
import os
import sys
from abc import ABC, abstractmethod

multiRAG_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, multiRAG_dir)
from multiRAG import MultiRAG
from ClassAssistant.LLMmodel import LLM_compus, LLM_psychology, LLM_paper, LLM_fitness


from Utils.Path import (
    PAPER_DOCS_DIR, CAMPUS_DOCS_DIR, FITNESS_DOCS_DIR, PSYCHOLOGY_DOCS_DIR,
    PAPER_INDEX_DIR, CAMPUS_INDEX_DIR, FITNESS_INDEX_DIR, PSYCHOLOGY_INDEX_DIR,
    ALL_PROCESSED_IMAGES_DIR, CAMPUS_IMAGES_DIR, PAPER_IMAGES_DIR, FITNESS_IMAGES_DIR, PSYCHOLOGY_IMAGES_DIR,
    CAMPUS_PROCESSED_EXTRACTED_IMAGES, PSYCHOLOGY_PROCESSED_EXTRACTED_IMAGES,
    CAMPUS_EXTRACTED_IMAGES_JSON, PSYCHOLOGY_EXTRACTED_IMAGES_JSON,
    CAMPUS_IMAGES_PATH, PSYCHOLOGY_IMAGES_PATH,
    CAMPUS_IMAGES_MAPPING_PATH, PSYCHOLOGY_IMAGES_MAPPING_PATH
)

APP_ID = "c2affdebf6664d438a4043216ee15dea"
apiKey = "sk-08331d936d254185aa67adff58bfd5eb"

class BaseAssistant(ABC):
    """所有助手的基类，提供通用功能"""
    
    def __init__(self, app_id, session_id, scene, llm_class, debug=False, use_fallback=True):
        # 从环境变量获取 API 配置
        self.app_id = app_id or os.getenv(f"APP_ID_{scene.upper()}", APP_ID)
        self.api_key = os.getenv("BAILIAN_API_KEY", apiKey)
        self.session_id = session_id
        self.multi_rag = MultiRAG(scene=scene)
        self.llm_instance = llm_class(self.app_id)
        self.debug = debug
        self.use_fallback = use_fallback
        
        if self.debug:
            print(f"{self.__class__.__name__} 配置:")
            print(f"  App ID: {self.app_id}")
            print(f"  Scene: {scene}")
            print(f"  Session ID: {self.session_id}")
            print(f"  Debug: {self.debug}")
            print(f"  Use Fallback: {self.use_fallback}")
        
        print(f"{self.__class__.__name__} MultiRAG 系统初始化完成")

    def _extract_text_from_chunk(self, chunk):
        """从chunk中提取文本内容的辅助方法"""
        # 方法1: 从output.text提取
        if hasattr(chunk, 'output') and hasattr(chunk.output, 'text') and chunk.output.text:
            return chunk.output.text
        
        # 方法2: 从output.choices提取
        if hasattr(chunk, 'output') and hasattr(chunk.output, 'choices'):
            for choice in chunk.output.choices:
                if hasattr(choice, 'text') and choice.text:
                    return choice.text
        
        # 方法3: 直接text属性
        if hasattr(chunk, 'text') and chunk.text:
            return chunk.text
        
        # 方法4: 尝试JSON解析
        try:
            if hasattr(chunk, '__dict__'):
                chunk_dict = chunk.__dict__
                if 'output' in chunk_dict and 'text' in chunk_dict['output']:
                    return chunk_dict['output']['text']
        except:
            pass
        
        return None

    def _call_llm_stream_simple(self, prompt):
        """简化的LLM流式调用方法 - 使用字典访问方式"""
        try:
            if self.debug:
                print(f"{self.__class__.__name__}: 开始调用LLM API...")
                print(f"Prompt预览: {prompt[:200]}...")
            
            response = Application.call(
                api_key=self.api_key,
                app_id=self.app_id,
                prompt=prompt,
                session_id=self.session_id,
                stream=True
            )
            
            full_text = ""
            for chunk in response:
                if self.debug:
                    print(f"原始chunk类型: {type(chunk)}")
                    print(f"原始chunk内容: {chunk}")
                
                # 使用字典方式访问响应数据
                text_content = ""
                
                try:
                    # 方法1: 尝试直接访问字典键
                    if isinstance(chunk, dict):
                        text_content = chunk.get('output', {}).get('text', '') or chunk.get('text', '')
                    else:
                        # 方法2: 尝试使用 getattr 或字典方式访问
                        try:
                            # 先尝试作为字典访问
                            text_content = chunk.get('output', {}).get('text', '') or chunk.get('text', '')
                        except:
                            # 再尝试作为对象访问
                            if hasattr(chunk, 'output') and hasattr(chunk.output, 'text'):
                                text_content = chunk.output.text
                            elif hasattr(chunk, 'text'):
                                text_content = chunk.text
                except Exception as e:
                    if self.debug:
                        print(f"提取文本时出错: {e}")
                
                # 如果以上方法都失败，尝试字符串转换
                if not text_content:
                    try:
                        chunk_str = str(chunk)
                        # 尝试从字符串中提取文本
                        import re
                        # 查找文本内容
                        if 'text' in chunk_str:
                            # 尝试匹配JSON格式
                            match = re.search(r'"text":\s*"([^"]*)"', chunk_str)
                            if match:
                                text_content = match.group(1)
                            else:
                                # 如果没有匹配到，使用整个字符串（去除特殊字符）
                                text_content = re.sub(r'[^\w\s\u4e00-\u9fff，。！？]', '', chunk_str)
                    except:
                        text_content = ""
                
                # 清理文本内容
                text_content = text_content.strip() if text_content else ""
                
                if text_content:
                    # 只返回新增的文本
                    if full_text and text_content.startswith(full_text):
                        new_text = text_content[len(full_text):]
                        if new_text.strip():
                            yield new_text
                            full_text = text_content
                    else:
                        yield text_content
                        full_text = text_content
                
                if self.debug and text_content:
                    print(f"提取到的文本: '{text_content}'")
            
            if self.debug:
                print(f"{self.__class__.__name__}: 流式调用完成，总文本长度: {len(full_text)}")
                
            # 如果最终没有生成任何文本，抛出异常
            if not full_text.strip():
                raise Exception("LLM API调用成功但返回了空内容")
        
        except Exception as e:
            error_message = f"{self.__class__.__name__}调用LLM时发生异常: {str(e)}"
            print(error_message)
            
            # 提供更详细的错误信息
            if self.debug:
                import traceback
                print("详细错误信息:")
                traceback.print_exc()
            
            # 重新抛出异常，让上层处理
            raise Exception(f"LLM调用失败: {str(e)}")
    
    def _generate_local_fallback_answer(self, prompt):
        """生成本地备用回答"""
        try:
            # 从prompt中提取关键信息
            if "用户问题:" in prompt and "背景知识:" in prompt:
                # 提取用户问题
                question_part = prompt.split("用户问题:")[1].split("背景知识:")[0].strip()
                # 提取背景知识
                knowledge_part = prompt.split("背景知识:")[1].split("回答要求:")[0].strip()
                
                # 简单的本地回答生成
                answer_parts = []
                answer_parts.append(f"关于您的问题'{question_part}'，我根据现有资料为您提供以下信息：")
                
                # 使用背景知识的前几段
                knowledge_lines = knowledge_part.split('\n\n')
                for i, line in enumerate(knowledge_lines[:3]):
                    if line.strip() and len(line.strip()) > 10:  # 过滤空行和过短的行
                        answer_parts.append(f"{line.strip()}")
                
                answer_parts.append("\n[注：当前使用本地备用模式，建议联系管理员处理API服务问题]")
                
                return '\n'.join(answer_parts)
            else:
                return "抱歉，当前AI服务暂时不可用，请稍后重试或联系管理员。"
                
        except Exception as e:
            return f"当前服务暂时不可用，请稍后重试。错误详情：{str(e)}"
    
    def _call_llm_stream_advanced(self, prompt):
        """高级LLM流式调用方法 - 修复版本"""
        try:
            if self.debug:
                print(f"{self.__class__.__name__}: 开始调用LLM API (高级版)...")
            
            response = Application.call(
                api_key=self.api_key,
                app_id=self.app_id,
                prompt=prompt,
                session_id=self.session_id,
                stream=True
            )
            
            chunk_count = 0
            valid_chunk_count = 0
            full_response_text = ""
            
            for chunk in response:
                chunk_count += 1
                
                # 使用 content 而不是 text
                text_content = ""
                try:
                    # 优先使用 content 属性
                    if hasattr(chunk, 'content') and chunk.content:
                        text_content = chunk.content
                    elif hasattr(chunk, 'output') and hasattr(chunk.output, 'content') and chunk.output.content:
                        text_content = chunk.output.content
                except Exception as e:
                    if self.debug:
                        print(f"提取文本时出错: {e}")
                
                if self.debug:
                    print(f"Chunk {chunk_count}: '{text_content}'")
                
                if text_content:
                    if full_response_text and text_content.startswith(full_response_text):
                        new_content = text_content[len(full_response_text):]
                        if new_content.strip():
                            yield new_content
                            valid_chunk_count += 1
                            full_response_text = text_content
                    else:
                        yield text_content
                        valid_chunk_count += 1
                        full_response_text = text_content
                else:
                    if self.debug:
                        print("忽略空chunk")
            
            if self.debug:
                print(f"{self.__class__.__name__}: 成功处理 {chunk_count} 个chunk，其中 {valid_chunk_count} 个有效")
        
        except Exception as e:
            error_message = f"{self.__class__.__name__}调用LLM时发生异常: {str(e)}"
            print(error_message)
            if self.debug:
                import traceback
                traceback.print_exc()
            raise Exception(f"LLM调用失败: {str(e)}")

    def _process_retrieval_results(self, results, image_map, image_output_dir):
        """处理检索结果，包括文本和图片 - 修复版本"""
        text_chunks = []
        images = []

        for result in results:
            document = result.get('document', '')
            source = result.get('source', '')
            result_type = result.get('type', 0)
            score = result.get('score', 0)
            doc_id = result.get('id', '')
            full_path = result.get('full_path', '')  # 获取完整路径

            # 判断是否是图片
            is_image = result_type == 1

            if is_image and document:
                # 从图片映射中获取详细信息
                img_info = image_map.get(doc_id, {})
                actual_image_path = img_info.get('image_path', source)
                # 优先使用 full_path，其次使用 actual_image_path
                final_image_path = full_path or actual_image_path
                
                # 确保是绝对路径
                if final_image_path and not os.path.isabs(final_image_path):
                    # 如果是相对路径，转换为绝对路径
                    final_image_path = os.path.abspath(final_image_path)
                
                image_filename = img_info.get('image_filename', '')
                enhanced_description = img_info.get('enhanced_description', document)
        
                image_info = {
                    'description': enhanced_description[:100] + '...' if len(enhanced_description) > 100 else enhanced_description,
                    'source': final_image_path,  # 使用最终确定的路径
                    'filename': image_filename,
                    'score': score,
                    'type': 'image',
                    'absolute_path': final_image_path,  # 添加绝对路径字段
                    'exists': os.path.exists(final_image_path) if final_image_path else False
                }
        
                # 检查图片文件是否存在
                if final_image_path and os.path.exists(final_image_path):
                    image_info['status'] = 'exists'
                    images.append(image_info)
                    text_chunks.append(f"[图片] {enhanced_description}")
                    if self.debug:
                        print(f"{self.__class__.__name__}: 找到有效图片: {final_image_path}")
                elif final_image_path:
                    image_info['status'] = 'missing'
                    images.append(image_info)
                    text_chunks.append(f"[图片-文件缺失] {enhanced_description}")
                    if self.debug:
                        print(f"{self.__class__.__name__}: 图片文件不存在: {final_image_path}")
                else:
                    image_info['status'] = 'no_path'
                    images.append(image_info)
                    text_chunks.append(f"[图片-无路径] {enhanced_description}")
                    if self.debug:
                        print(f"{self.__class__.__name__}: 图片无路径信息: {doc_id}")
            else:
                # 文本内容
                text_chunks.append(document)

        return text_chunks, images

    def _enhance_chunks(self, text_chunks, image_info):
        """根据图片信息增强文本片段 - 通用实现"""
        enhanced_chunks = text_chunks.copy()

        if image_info:
            existing_images = [img for img in image_info if img.get('status') == 'exists']
            if existing_images:
                image_instruction = f"\n注意：以下{self.__class__.__name__}相关的图片内容可供参考：\n"
                for i, img in enumerate(existing_images[:3], 1):
                    image_instruction += f"{i}. {img['description']}\n"
                enhanced_chunks.append(image_instruction)

        return enhanced_chunks

    def _create_empty_response(self, stream_mode, message):
        """创建空响应"""
        if stream_mode:
            return {"answer_generator": iter([message]), "images": [], "total_results": 0}
        else:
            return {"answer": message, "images": [], "total_results": 0}

    def _create_error_response(self, stream_mode, error_msg):
        """创建错误响应"""
        message = f"检索过程中发生错误: {error_msg}"
        if stream_mode:
            return {"answer_generator": iter([message]), "images": [], "total_results": 0}
        else:
            return {"answer": message, "images": [], "total_results": 0}

    def retrieve_and_answer(self, query, top_k=8, stream_mode=False):
        """检索并回答 - 简化版本，无索引检查"""
        print(f"{self.__class__.__name__}: 正在检索与问题相关的top-{top_k}片段...")

        try:
            # 直接检索相关文档，不进行索引检查
            results = self.multi_rag.retrieve(query, top_k)

            if not results:
                print(f"{self.__class__.__name__}: 未找到相关片段，使用通用知识回答")
                return self._fallback_answer(query, stream_mode)

            print(f"{self.__class__.__name__}: 找到 {len(results)} 个相关片段")

            # 获取图片映射和输出目录
            image_map = self.multi_rag._load_image_mapping()
            image_output_dir = self.multi_rag.image_output_dir

            # 处理检索结果
            text_chunks, images = self._process_retrieval_results(results, image_map, image_output_dir)
            
            if self.debug:
                print(f"{self.__class__.__name__}: 处理后的文本片段数量: {len(text_chunks)}")
                print(f"{self.__class__.__name__}: 处理后的图片数量: {len(images)}")
                for i, chunk in enumerate(text_chunks[:3]):  # 显示前3个文本片段
                    print(f"  文本片段{i+1}: {chunk[:100]}...")

            # 生成最终回答
            return self._generate_final_answer(query, text_chunks, images, stream_mode)

        except Exception as e:
            print(f"{self.__class__.__name__}: 检索过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_answer(query, stream_mode)
        
    def _generate_final_answer(self, query, text_chunks, images, stream_mode=False):
        """使用处理后的文本块和图片生成最终回答"""
        # 增强文本片段
        enhanced_chunks = self._enhance_chunks(text_chunks, images)
        total_results = len(enhanced_chunks)
    
        # 根据是否使用备用方案选择生成方式
        if self.use_fallback:
            return self._generate_response_simple(query, enhanced_chunks, images, total_results, stream_mode)
        else:
            return self._generate_response_advanced(query, enhanced_chunks, images, total_results, stream_mode)

    def _fallback_answer(self, query, stream_mode):
        """后备回答方法"""
        if stream_mode:
            return {
            "answer_generator": iter([f"关于'{query}'，我暂时没有找到相关的专门资料，但根据我的理解：\n\n这是一个常见的问题，建议您咨询相关部门或查看官方网站获取最新信息。"]),
            "images": []
        }
        else:
            return {
                "answer": f"关于'{query}'，我暂时没有找到相关的专门资料，但根据我的理解：\n\n这是一个常见的问题，建议您咨询相关部门或查看官方网站获取最新信息。",
                "images": []
            }
        
    def _generate_response_simple(self, query, enhanced_chunks, images, total_results, stream_mode):
        """使用简化方案生成响应 - 修复版本"""
        separator = "\n\n"
        system_prompt = self.get_system_prompt()
        response_requirements = self.get_response_requirements()
        
        prompt = f"""{system_prompt}


    请根据用户的问题和下面的背景知识进行回答。

    用户问题: {query}

    背景知识:
    {separator.join(enhanced_chunks)}

    {response_requirements}

    请开始你的回答：
    """

        if self.debug:
            print(f"{self.__class__.__name__}: 生成的Prompt长度: {len(prompt)}")
            print(f"{self.__class__.__name__}: 背景知识片段数量: {len(enhanced_chunks)}")

        if stream_mode:
            return {
                "answer_generator": self._call_llm_stream_simple(prompt),
                "images": [img for img in images if img.get('status') == 'exists'],
                "total_results": total_results
            }
        else:
            if self.debug:
                print(f"{self.__class__.__name__}: 开始非流式调用LLM...")
            
            answer_chunks = list(self._call_llm_stream_simple(prompt))
            answer = "".join(answer_chunks)
            
            if self.debug:
                print(f"{self.__class__.__name__}: 生成的回答长度: {len(answer)}")
                if answer:
                    print(f"{self.__class__.__name__}: 回答预览: {answer[:200]}...")
            
            return {
                "answer": answer,
                "images": [img for img in images if img.get('status') == 'exists'],
                "total_results": total_results
            }
        
    def _generate_response_advanced(self, query, enhanced_chunks, images, total_results, stream_mode):
        """使用高级方案生成响应"""
        separator = "\n\n"
        system_prompt = self.get_system_prompt()
        response_requirements = self.get_response_requirements()
        # 构建包含真实图片路径的背景知识
        enhanced_background = enhanced_chunks.copy()
        
        # 如果有图片，将图片路径信息添加到背景知识中
        if images:
            image_info = "\n\n相关图片信息:\n"
            for i, img in enumerate(images, 1):
                if img.get('status') == 'exists' and img.get('source'):
                    img_path = img.get('source', '')
                    img_filename = img.get('filename', os.path.basename(img_path))
                    img_description = img.get('description', '')[:100] + '...' if len(img.get('description', '')) > 100 else img.get('description', '')
                    
                    image_info += f"图片{i}: {img_filename}\n"
                    image_info += f"路径: {img_path}\n"
                    image_info += f"描述: {img_description}\n\n"
            
            enhanced_background.append(image_info)
        
        prompt = f"""{system_prompt}


请根据用户的问题和下面的背景知识进行回答。

用户问题: {query}

背景知识:
{separator.join(enhanced_chunks)}

{response_requirements}

请开始你的回答：
"""

        if stream_mode:
            return {
                "answer_generator": self._call_llm_stream_advanced(prompt),
                "images": [img for img in images if img.get('status') == 'exists'],
                "total_results": total_results
            }
        else:
            answer_chunks = list(self._call_llm_stream_advanced(prompt))
            answer = "".join(answer_chunks)
            return {
                "answer": answer,
                "images": [img for img in images if img.get('status') == 'exists'],
                "total_results": total_results
            }

    def _generate_enhanced_local_answer(self, query, text_chunks, images):
        """基于检索内容生成增强的本地回答"""
        try:
            # 构建基于检索内容的回答
            answer_parts = []
            answer_parts.append(f"关于'{query}'，我根据本地知识库为您提供以下信息：\n")
            
            # 添加最重要的文本片段
            for i, chunk in enumerate(text_chunks[:5]):
                if len(chunk.strip()) > 20:  # 只添加有实质内容的片段
                    # 清理文本
                    clean_chunk = chunk.replace('[图片]', '').replace('[图片-文件缺失]', '').strip()
                    if clean_chunk:
                        answer_parts.append(f"{clean_chunk}")
            
            # 添加图片信息
            if images:
                answer_parts.append("\n相关图片信息：")
                existing_images = [img for img in images if img.get('status') == 'exists']
                for i, img in enumerate(existing_images[:3]):
                    answer_parts.append(f"图片{i+1}: {img.get('description', '相关示意图')}")
            
            answer_parts.append("\n\n[当前使用本地知识库模式，AI增强功能暂不可用]")
            
            return '\n'.join(answer_parts)
            
        except Exception as e:
            return f"基于本地知识库生成回答时出错: {str(e)}"

    @abstractmethod
    def start_service(self):
        """启动服务 - 子类必须实现"""
        pass

    @abstractmethod
    def get_system_prompt(self):
        """获取系统提示词 - 子类必须实现"""
        pass

    @abstractmethod
    def get_image_keywords(self):
        """获取图片关键词 - 子类必须实现"""
        pass

    @abstractmethod
    def get_response_requirements(self):
        """获取回答要求 - 子类必须实现"""
        pass
    def close(self):
        """清理资源"""
        try:
            if hasattr(self, 'multi_rag'):
                # 如果有需要清理的 MultiRAG 资源
                pass
            print(f"{self.__class__.__name__} 资源已清理")
        except Exception as e:
            print(f"清理资源时出错: {e}")

    def __del__(self):
        """析构函数"""
        self.close()
# 具体的助手类实现
class CampusAssistant(BaseAssistant):
    def __init__(self, app_id=None, debug=False, use_fallback=True, **kwargs):
        # 确保 app_id 有合理的默认值
        actual_app_id = app_id or os.getenv('APP_ID_CAMPUS')
        if not actual_app_id:
            raise ValueError("CampusAssistant: 必须提供 app_id 或设置 APP_ID_CAMPUS 环境变量")
        
        super().__init__(actual_app_id, "campus_session", "campus", LLM_compus, 
                        debug=debug, use_fallback=use_fallback, **kwargs)
        
        # 验证 MultiRAG 初始化
        if hasattr(self.multi_rag, 'index_path'):
            print(f"✅ CampusAssistant MultiRAG 初始化成功")
            print(f"   索引路径: {self.multi_rag.index_path}")
        else:
            print("❌ CampusAssistant MultiRAG 初始化异常")

    def start_service(self):
        """启动服务 - 简化版本，无索引检查"""
        try:
            # 直接返回成功，不进行索引检查
            if self.debug:
                print("CampusAssistant 服务启动成功")
            return "校园助手启动成功"
        except Exception as e:
            error_msg = f"CampusAssistant 启动失败: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)

    def get_system_prompt(self):
        return self.llm_instance.get_stream_system_prompt()

    def get_image_keywords(self):
        return ['图片', '图像', '图示', '照片', '图表', '校园', '地图', '建筑']

    def get_response_requirements(self):
        return """回答要求：
1. 模仿人类口吻，友好自然地进行分段说明。
2. 将完整的回答分成3到5段，每段都要有实质内容，段与段之间要在语义和逻辑上相互承接，段落之间必须用 `[NEW_PARAGRAPH]` 分隔。
3. 如果背景知识中包含图片信息，请在回答中具体引用图片内容，使用图片的实际文件名而不是占位符。
4. 回答要具体、实用，提供可操作的信息，避免模糊的表述。
5. 若用户问题与背景知识无关，则用通用知识解决问题。"""


class PsychologyAssistant(BaseAssistant):
    def __init__(self, app_id=None, debug=False, use_fallback=True, **kwargs):
        super().__init__(app_id, "psychology_session", "psychology", LLM_psychology, 
                        debug=debug, use_fallback=use_fallback, **kwargs)

    def start_service(self):
        """启动服务 - 简化版本"""
        return "心理学助手启动成功"

    def get_system_prompt(self):
        return self.llm_instance.get_stream_system_prompt()

    def get_image_keywords(self):
        return ['图片', '图像', '图示', '照片', '图表', '心理', '情绪', '认知', '大脑', '心理测试']

    def get_response_requirements(self):
        return """回答要求：
1. 用类似人类和朋友聊天的语言进行回答，不要太一板一眼，温暖、专业、富有同理心。
2. 将完整的回答分成3到5段，每段都要有实质内容，段与段之间要在语义和逻辑上相互承接，段落之间必须用 `[NEW_PARAGRAPH]` 分隔。
3. 如果背景知识中包含图片信息，请在回答中具体引用图片内容。
4. 严守边界，不做诊断：明确声明非医疗身份，禁止提供任何医学/心理诊断或治疗方案。
5. 回答要具体、有深度，避免模糊的表述。"""


class PaperAssistant(BaseAssistant):
    # def __init__(self, app_id=None, debug=False, use_fallback=True, **kwargs):
    #     # 只继承 BaseAssistant，不多重继承
    #     super().__init__(app_id, "paper_session", "paper", LLM_paper, **kwargs)
    #     # 手动设置 LLM 实例
    #     self.llm_instance = LLM_paper(self.app_id)

    def start_service(self):
        return "论文助手启动成功"

    def get_system_prompt(self):
        return self.llm_instance.get_stream_system_prompt()

    def get_image_keywords(self):
        return ['图片', '图像', '图示', '照片', '图表', '数据', '可视化', '实验', '图表', '论文', '研究']

    def get_response_requirements(self):
        return """回答要求：
1. 用严谨、学术、专业的语言进行回答，保持论文写作风格。
2. 将完整的回答分成3到5段，段与段之间要在语义和逻辑上相互承接，段落之间必须用 `[NEW_PARAGRAPH]` 分隔。
3. 如果背景知识中包含论文图表、数据可视化或实验图示，请在回答中适当引用。
4. 引用图表时，直接使用提供的图表地址，格式：[具体路径]，无需任何前缀或后缀。
5. 若用户问题与背景知识无关，则用通用学术知识解决问题。
6. 保持学术严谨性，同时要清晰易懂。"""


class FitnessAssistant(BaseAssistant):
    # def __init__(self, app_id=None, debug=False, use_fallback=True, **kwargs):
    #     # 只继承 BaseAssistant，不多重继承
    #     super().__init__(app_id, "fitness_session", "fitness", LLM_fitness, **kwargs)
    #     # 手动设置 LLM 实例
    #     self.llm_instance = LLM_fitness(self.app_id)

    def start_service(self):
        return "健康饮食助手启动成功"

    def get_system_prompt(self):
        return self.llm_instance.get_stream_system_prompt()

    def get_image_keywords(self):
        return ['图片', '图像', '图示', '照片', '图表', '健身', '运动', '营养', '饮食', '动作', '解剖']

    def get_response_requirements(self):
        return """回答要求：
1. 用鼓励、专业、实用的语言进行回答，保持健身教练风格。
2. 将完整的回答分成3到5段，段与段之间要在语义和逻辑上相互承接，段落之间必须用 `[NEW_PARAGRAPH]` 分隔。
3. 如果背景知识中包含健身动作图示、营养图表或解剖图示，请在回答中适当引用。
4. 引用图示时，直接使用提供的图示地址，格式：[具体路径]，无需任何前缀或后缀。
5. 若用户问题与背景知识无关，则用通用健身营养知识解决问题。
6. 保持专业性，同时要鼓励和支持用户。"""