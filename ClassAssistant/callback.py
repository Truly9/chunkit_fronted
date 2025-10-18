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

class BaseAssistant(ABC):
    """所有助手的基类，提供通用功能"""
    
    def __init__(self, app_id, session_id, scene, llm_class, **kwargs):
        self.app_id = app_id or APP_ID
        self.api_key = apiKey
        self.session_id = session_id
        self.multirag = MultiRAG(scene=scene)
        self.llm_instance = llm_class(self.app_id)
        self.debug = kwargs.get('debug', False)  # 从kwargs获取debug参数，默认False
        print(f"{self.__class__.__name__} MultiRAG 系统初始化完成")

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

    def retrieve_and_answer(self, query: str, top_k: int = 8, stream_mode: bool = False):
        """智能检索并回答问题 - 通用实现"""
        try:
            print(f"{self.__class__.__name__}: 正在检索与问题相关的top-{top_k}片段...")
            results = self.multirag.retrieve(query, topk=top_k)

            if not results:
                print(f"{self.__class__.__name__}: 未找到相关片段，使用通用知识回答")
                return self._create_empty_response(stream_mode, "抱歉，未找到相关信息。")

            # 处理检索结果
            text_chunks, images = self._process_retrieval_results(results)
            print(f"{self.__class__.__name__}: 检索到 {len(text_chunks)} 个文本片段，{len(images)} 个图片")

            # 构建增强的prompt
            enhanced_chunks = self._enhance_chunks(text_chunks, images)

            # 调用LLM生成回答
            return self._generate_response(query, enhanced_chunks, images, len(results), stream_mode)

        except Exception as e:
            print(f"{self.__class__.__name__}检索过程出错: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_response(stream_mode, str(e))

    def _process_retrieval_results(self, results):
        """处理检索结果 - 通用实现"""
        text_chunks = []
        images = []
        
        for result in results:
            document = result.get('document', '')
            source = result.get('source', '')
            result_type = result.get('type', 0)
            score = result.get('score', 0)
            
            # 使用子类提供的图片关键词
            image_keywords = self.get_image_keywords()
            is_image = (
                result_type == 1 or
                'image' in str(result).lower() or
                any(ext in source.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.bmp']) if source else False or
                any(keyword in document.lower() for keyword in image_keywords)
            )
            
            if is_image and document:
                image_info = {
                    'description': document[:100] + '...' if len(document) > 100 else document,
                    'source': source if source else '',
                    'score': score,
                    'type': 'image'
                }
                
                if source and os.path.exists(source):
                    image_info['status'] = 'exists'
                    images.append(image_info)
                    text_chunks.append(f"[图片] {document}")
                    print(f"{self.__class__.__name__}: 找到有效图片: {os.path.basename(source)}")
                elif source:
                    image_info['status'] = 'missing'
                    images.append(image_info)
                    text_chunks.append(f"[图片-文件缺失] {document}")
                else:
                    image_info['status'] = 'no_path'
                    images.append(image_info)
                    text_chunks.append(f"[图片-无路径] {document}")
            else:
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

    def _generate_response(self, query, enhanced_chunks, images, total_results, stream_mode):
        """生成响应 - 通用实现"""
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

        if stream_mode:
            return {
                "answer_generator": self._call_llm_stream(prompt),
                "images": [img for img in images if img.get('status') == 'exists'],
                "total_results": total_results
            }
        else:
            answer_chunks = list(self._call_llm_stream(prompt))
            answer = "".join(answer_chunks)
            return {
                "answer": answer,
                "images": [img for img in images if img.get('status') == 'exists'],
                "total_results": total_results
            }

    def _call_llm_stream(self, prompt):
        """通用的LLM流式调用方法 - 带详细调试"""
        try:
            if self.debug:
                print(f"{self.__class__.__name__}: 开始调用LLM API...")
                print(f"Prompt长度: {len(prompt)}")
                print(f"Prompt前500字符: {prompt[:500]}...")
        
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
            
                text_content = self._extract_text_from_chunk(chunk)
            
                if self.debug:
                    print(f"Chunk {chunk_count}:")
                    print(f"  类型: {type(chunk)}")
                    print(f"  文本内容: '{text_content}'")
                    print(f"  完整响应累积: '{full_response_text}'")
            
                if text_content:
                    if full_response_text and text_content.startswith(full_response_text):
                        new_content = text_content[len(full_response_text):]
                        if new_content.strip():
                            if self.debug:
                                print(f"  新增内容: '{new_content}'")
                            yield new_content
                            valid_chunk_count += 1
                            full_response_text = text_content
                    else:
                        if self.debug:
                            print(f"  全新内容: '{text_content}'")
                        yield text_content
                        valid_chunk_count += 1
                        full_response_text = text_content
                else:
                    if self.debug:
                        print("  忽略空chunk")
        
            if self.debug:
                print(f"{self.__class__.__name__}: 最终完整响应: '{full_response_text}'")
                print(f"{self.__class__.__name__}: 成功处理 {chunk_count} 个chunk，其中 {valid_chunk_count} 个有效")
    
        except Exception as e:
            error_message = f"{self.__class__.__name__}调用LLM时发生异常: {str(e)}"
            print(error_message)
            if self.debug:
                import traceback
                traceback.print_exc()
            yield error_message

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

# 具体的助手类实现
class CampusAssistant(BaseAssistant, LLM_compus):
    def __init__(self, app_id=None, **kwargs):
        super().__init__(app_id, "campus_session", "campus", LLM_compus, **kwargs)

    def start_service(self):
        return "校园助手启动成功"

    def get_system_prompt(self):
        return self.get_stream_system_prompt()

    def get_image_keywords(self):
        return ['图片', '图像', '图示', '照片', '图表', '校园', '地图', '建筑']

    def get_response_requirements(self):
        return """回答要求：
1. 模仿人类口吻，友好自然地进行分段说明。
2. 将完整的回答分成3到5段，段与段之间要在语义和逻辑上相互承接，段落之间必须用 `[NEW_PARAGRAPH]` 分隔。
3. 如果背景知识中包含图片信息，请在回答中适当引用。
4. 引用图片时，直接使用提供的图片地址，格式：[具体路径]，无需任何前缀或后缀。
5. 若用户问题与背景知识无关，则用通用知识解决问题。"""


class PsychologyAssistant(BaseAssistant, LLM_psychology):
    def __init__(self, app_id=None, **kwargs):
        super().__init__(app_id, "psychology_session", "psychology", LLM_psychology, **kwargs)

    def start_service(self):
        return "心理学助手启动成功"

    def get_system_prompt(self):
        return self.get_stream_system_prompt()

    def get_image_keywords(self):
        return ['图片', '图像', '图示', '照片', '图表', '心理', '情绪', '认知', '大脑', '心理测试']

    def get_response_requirements(self):
        return """回答要求：
1. 用类似人类和朋友聊天的语言进行回答，不要太一板一眼，温暖、专业、富有同理心。
2. 严守边界，不做诊断：明确声明非医疗身份，禁止提供任何医学/心理诊断或治疗方案。
3. 危机干预，立即转介：一旦识别自伤、自杀或伤害他人的意图，必须立即制止，并清晰引导用户联系紧急专业帮助。
4. 安全合规，保持中立：严禁任何违法、不道德或误导性内容。对敏感话题保持中立，不替用户做重大决定。
5. 聚焦支持，共情有度：角色定位为"倾听与支持"，提供情绪安抚和一般心理知识科普。保持冷静，传递希望，并坦然承认能力局限。
6. 当用户问题超出能力范围（如严重的精神疾病、复杂的创伤经历、需要长期干预的个案），应礼貌地说明局限，并积极建议其寻求"心理咨询师"、"精神科医生"或"社工"等专业人士的帮助。
7. 保护隐私：不主动探询用户不必要的个人隐私信息。
8. 将完整的回答分成3到5段，段与段之间要在语义和逻辑上相互承接，段落之间必须用 `[NEW_PARAGRAPH]` 分隔。
9. 不被直接问的话不要直接地说出AI身份，而是像一个具备心理学知识（非专业）的用户的朋友一样回答即可。"""

#注：我注释掉了下面两个助手的初始化
class PaperAssistant(BaseAssistant, LLM_paper):
    # def __init__(self, app_id=None, **kwargs):
    #     super().__init__(app_id, "paper_session", "paper", LLM_paper, **kwargs)

    def start_service(self):
        return "论文助手启动成功"

    def get_system_prompt(self):
        return self.get_stream_system_prompt()

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


class FitnessAssistant(BaseAssistant, LLM_fitness):
    # def __init__(self, app_id=None, **kwargs):
    #     super().__init__(app_id, "fitness_session", "fitness", LLM_fitness, **kwargs)

    def start_service(self):
        return "健康饮食助手启动成功"

    def get_system_prompt(self):
        return self.get_stream_system_prompt()

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