from dashscope import Application
from http import HTTPStatus
import os
import json
from multiRAG import MultiRAG

# 从Path文件里面引入知识库文件地址,索引文件的地址
from Utils.Path import (
    PAPER_DOCS_DIR, CAMPUS_DOCS_DIR, FITNESS_DOCS_DIR, PSYCHOLOGY_DOCS_DIR,
    PAPER_INDEX_DIR, CAMPUS_INDEX_DIR, FITNESS_INDEX_DIR, PSYCHOLOGY_INDEX_DIR,
    ALL_PROCESSED_IMAGES_DIR, CAMPUS_IMAGES_DIR, PAPER_IMAGES_DIR, FITNESS_IMAGES_DIR, PSYCHOLOGY_IMAGES_DIR,
    CAMPUS_PROCESSED_EXTRACTED_IMAGES, PSYCHOLOGY_PROCESSED_EXTRACTED_IMAGES,
    CAMPUS_EXTRACTED_IMAGES_JSON, PSYCHOLOGY_EXTRACTED_IMAGES_JSON,
    CAMPUS_IMAGES_PATH, PSYCHOLOGY_IMAGES_PATH,
    CAMPUS_IMAGES_MAPPING_PATH, PSYCHOLOGY_IMAGES_MAPPING_PATH
)


from ClassAssistant.LLMmodel import LLM_compus, LLM_psychology, LLM_paper, LLM_fitness

APP_ID = "c2affdebf6664d438a4043216ee15dea"
apiKey = "sk-f89e754d6cff4f31a25f609e82b3bce1"

class CampusAssistant(LLM_compus):
    def __init__(self, app_id=None):
        super().__init__(app_id or APP_ID)
        self.session_id = "campus_session"
        # 只需传递场景参数
        self.multirag = MultiRAG(scene="campus")
        print("校园助手 MultiRAG 系统初始化完成")

    def start_LLM(self):
        """
        启动校园助手服务
        """
        return "校园助手 LLM model started successfully"

    def retrieve_and_answer(self, query: str, top_k: int = 8):
        """
        智能检索并回答问题 - 校园助手专用

        Args:
            query (str): 用户问题
            top_k (int): 检索的片段数量

        Yields:
            str: 生成的文本段落
        """
        try:
            # 1. 使用MultiRAG检索相关片段
            print(f"校园助手: 正在检索与问题相关的top-{top_k}片段...")
            results = self.multirag.retrieve(query, topk=top_k)

            if not results:
                print("校园助手: 未找到相关片段，使用通用知识回答")
                yield from self.call_llm_stream(query, [])
                return

            # 2. 处理检索结果
            text_chunks = []
            image_info = []

            for result in results:
                result_type = result.get('type', 0)
                document = result.get('document', '')
                source = result.get('source', '')

                if result_type == 1:
                    if source and source != "":
                        image_info.append({
                            'description': document,
                            'path': source,
                            'score': 1.0
                        })
                        text_chunks.append(f"[图片内容] {document} [图片地址: {source}]")
                    else:
                        text_chunks.append(f"[图片内容] {document}")
                else:
                    text_chunks.append(document)

            print(f"校园助手: 检索到 {len(text_chunks)} 个文本片段，{len(image_info)} 个图片")

            # 3. 构建增强的prompt
            enhanced_chunks = self._enhance_chunks_with_images(text_chunks, image_info)

            # 4. 调用父类的流式生成方法
            yield from self.call_llm_stream(query, enhanced_chunks)

        except Exception as e:
            print(f"校园助手检索过程出错: {e}")
            import traceback
            traceback.print_exc()
            yield from self.call_llm_stream(query, [])

    def _enhance_chunks_with_images(self, text_chunks, image_info):
        """
        根据图片信息增强文本片段
        """
        enhanced_chunks = text_chunks.copy()

        if image_info:
            image_instruction = "\n注意：回答中如需引用图片，请直接使用图片地址，格式为：[具体路径]\n"
            enhanced_chunks.append(image_instruction)

            image_summary = "可用图片资源：\n"
            for i, img in enumerate(image_info[:3]):
                image_summary += f"{i + 1}. {img['description']} [地址: {img['path']}]\n"
            enhanced_chunks.append(image_summary)

        return enhanced_chunks

    def call_llm_stream(self, query, list):
        """
        重写父类的流式生成方法，添加校园助手专用的提示词增强
        """
        separator = "\n\n"
        # 使用父类的系统提示词，并添加校园专用增强
        system_prompt = self.get_stream_system_prompt()
        
        prompt = f"""{system_prompt}

请根据用户的问题和下面的背景知识进行回答。

用户问题: {query}

背景知识:
{separator.join(list)}

回答要求：
1. 模仿人类口吻，友好自然地进行分段说明。
2. 将完整的回答分成3到5段，段与段之间要在语义和逻辑上相互承接，段落之间必须用 `[NEW_PARAGRAPH]` 分隔。
3. 如果背景知识中包含图片信息（标注为[图片内容]或[图片地址]），请在回答中适当引用。
4. 引用图片时，直接使用提供的图片地址，格式：[具体路径]，无需任何前缀或后缀。
5. 若用户问题与背景知识无关，则用通用知识解决问题。

请开始你的回答：
"""

        # 使用父类的非流式调用逻辑
        full_response_text = ""
        try:
            response = Application.call(
                api_key=self.api_key,
                app_id=self.app_id,
                prompt=prompt,
                session_id=self.session_id,
                stream=False
            )
            if response.status_code == HTTPStatus.OK:
                request_id = response.request_id
                print(f"校园助手: 成功获取到回答，Request ID: {request_id}")
                full_response_text = response.output.text
            else:
                error_message = f'校园助手 API Error: {response.message}'
                print(error_message)
                yield error_message
                return

        except Exception as e:
            error_message = f"校园助手调用LLM时发生异常: {e}"
            print(error_message)
            yield error_message
            return

        # 根据分隔符切分段落并依次返回
        paragraphs = full_response_text.split('[NEW_PARAGRAPH]')
        for para in paragraphs:
            cleaned_para = para.strip()
            if cleaned_para:
                yield cleaned_para


class PsychologyAssistant(LLM_psychology):
    def __init__(self, app_id=None):
        super().__init__(app_id or APP_ID)
        self.session_id = "psychology_session"
        # 初始化MultiRAG系统 - 心理学场景
        self.multirag = MultiRAG(scene="psychology")
        print("心理助手 MultiRAG 系统初始化完成")

    def start_psychology(self):
        """启动心理学助手"""
        return "心理学助手启动成功"

    def retrieve_with_images(self, query: str, top_k: int = 8):
        """修复的检索方法 - 正确使用图片映射文件"""
        try:
            print(f"心理助手: 正在检索与问题相关的top-{top_k}片段...")
            
            # 1. 使用MultiRAG检索
            results = self.multirag.retrieve(query, topk=top_k)

            if not results:
                return {
                    "answer": "抱歉，没有找到相关信息。",
                    "images": [],
                    "total_results": 0
                }

            # 2. 处理检索结果
            text_chunks = []
            images = []

            for result in results:
                result_type = result.get('type', 0)
                document = result.get('document', '')
                source = result.get('source', '')
                score = result.get('score', 0)

                if result_type == 1:  # 图片类型
                    if source and source != "" and os.path.exists(source):
                        images.append({
                            'source': source,
                            'description': document[:100] + '...' if len(document) > 100 else document,
                            'score': score
                        })
                        text_chunks.append(f"[图片] {document}")
                        print(f"✅ 添加图片: {os.path.basename(source)}")
                    else:
                        text_chunks.append(f"[图片] {document}")
                        print(f"⚠️ 图片路径无效: {source}")
                else:
                    text_chunks.append(document)

            print(f"心理助手: 检索到 {len(text_chunks)} 个文本片段，{len(images)} 个图片")

            # 3. 如果图片数量不足，专门检索图片
            if len(images) < 1:
                print(f"心理助手: 图片数量不足，专门检索图片...")
                additional_images = self._retrieve_images_only(query, top_k=3)
                if additional_images:
                    print(f"心理助手: 专门检索找到 {len(additional_images)} 个额外图片")
                    images.extend(additional_images)
                    for img in additional_images:
                        text_chunks.append(f"[图片] {img['description']}")

            # 4. 构建增强的prompt
            enhanced_chunks = self._enhance_psychology_chunks(text_chunks, images)

            # 5. 调用LLM生成回答
            answer_generator = self.call_psychology_llm_stream(query, enhanced_chunks)
            answer = "".join(answer_generator)

            return {
                "answer": answer,
                "images": images,
                "total_results": len(results)
            }

        except Exception as e:
            print(f"心理助手检索过程出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"检索过程中发生错误: {str(e)}",
                "images": [],
                "total_results": 0
            }

    def _retrieve_images_only(self, query: str, top_k: int = 3):
        """专门检索图片"""
        try:
            # 使用更大的topk值专门检索图片
            image_results = self.multirag.retrieve(query, topk=top_k * 3)
            
            images = []
            for result in image_results:
                result_type = result.get('type', 0)
                if result_type == 1:  # 只处理图片类型
                    source = result.get('source', '')
                    document = result.get('document', '')
                    score = result.get('score', 0)
                    
                    if source and source != "" and os.path.exists(source):
                        images.append({
                            'source': source,
                            'description': document[:100] + '...' if len(document) > 100 else document,
                            'score': score
                        })
                        print(f"✅ 专门检索找到图片: {os.path.basename(source)}")
                        
                        if len(images) >= top_k:  # 达到目标数量就停止
                            break
            
            return images
            
        except Exception as e:
            print(f"专门检索图片时出错: {e}")
            return []
        
    def retrieve_and_answer(self, query: str, top_k: int = 5):
        """流式回答的兼容方法"""
        result = self.retrieve_with_images(query, top_k)
        yield result["answer"]

    def _enhance_psychology_chunks(self, text_chunks, image_info):
        """
        根据图片信息增强心理学文本片段
        """
        enhanced_chunks = text_chunks.copy()

        if image_info:
            image_instruction = "\n注意：回答中如需引用心理学相关的图示或案例图片，请直接使用图片地址，格式为：[具体路径]\n"
            enhanced_chunks.append(image_instruction)

            image_summary = "可用心理学图片资源：\n"
            for i, img in enumerate(image_info[:3]):
                image_summary += f"{i + 1}. {img['description']} [地址: {img['source']}]\n"
            enhanced_chunks.append(image_summary)

        return enhanced_chunks

    def debug_image_mapping():
        """调试图片映射文件"""
        mapping_file = str(PSYCHOLOGY_IMAGES_MAPPING_PATH)
    
        if not os.path.exists(mapping_file):
            print(f"❌ 图片映射文件不存在: {mapping_file}")
            return
    
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
    
        print(f"📊 图片映射文件调试信息:")
        print(f"  总图片数: {len(mapping)}")
    
        # 检查前5个图片的详细信息
        for i, (img_id, img_info) in enumerate(list(mapping.items())[:5]):
            image_path = img_info.get('image_path', '')
            exists = os.path.exists(image_path) if image_path else False
        
            print(f"\n{i+1}. {img_id}")
            print(f"   路径: {image_path} {'✅' if exists else '❌'}")
            print(f"   描述: {img_info.get('enhanced_description', '')[:100]}...")

    # 在适当的地方调用调试函数
    debug_image_mapping()

    def call_psychology_llm_stream(self, query, list):
        """
        心理助手专用的流式生成方法
        """
        separator = "\n\n"
        # 使用父类的心理学系统提示词
        system_prompt = self.get_stream_system_prompt()
        
        prompt = f"""{system_prompt}

请根据用户的问题和下面的背景知识进行回答。

用户问题: {query}

背景知识:
{separator.join(list)}

回答要求：
1. 用温暖、专业、富有同理心的语言进行回答。
2. 将完整的回答分成3到5段，段与段之间要在语义和逻辑上相互承接，段落之间必须用 `[NEW_PARAGRAPH]` 分隔。
3. 如果背景知识中包含心理学相关的图片信息（标注为[图片内容]或[图片地址]），请在回答中适当引用。
4. 引用图片时，直接使用提供的图片地址，格式：[具体路径]，无需任何前缀或后缀。
5. 若用户问题与背景知识无关，则用通用心理学知识解决问题。
6. 保持专业性，同时要温暖和有同理心。

请开始你的回答：
"""

        # 使用父类的非流式调用逻辑
        full_response_text = ""
        try:
            response = Application.call(
                api_key=self.api_key,
                app_id=self.app_id,
                prompt=prompt,
                session_id=self.session_id,
                stream=False
            )
            if response.status_code == HTTPStatus.OK:
                request_id = response.request_id
                print(f"心理助手: 成功获取到回答，Request ID: {request_id}")
                full_response_text = response.output.text
            else:
                error_message = f'心理助手 API Error: {response.message}'
                print(error_message)
                yield error_message
                return

        except Exception as e:
            error_message = f"心理助手调用LLM时发生异常: {e}"
            print(error_message)
            yield error_message
            return

        # 根据分隔符切分段落并依次返回
        paragraphs = full_response_text.split('[NEW_PARAGRAPH]')
        for para in paragraphs:
            cleaned_para = para.strip()
            if cleaned_para:
                yield cleaned_para

    def check_psychology_image_mapping():
        """检查心理学图片映射文件的内容"""
        mapping_file = str(PSYCHOLOGY_IMAGES_MAPPING_PATH)
    
        if not os.path.exists(mapping_file):
            print(f"❌ 图片映射文件不存在: {mapping_file}")
            return
    
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
    
        print(f"📊 心理学图片映射文件统计:")
        print(f"  总图片数: {len(mapping)}")
    
        # 检查前10个图片的描述
        print(f"\n🔍 前10个图片描述示例:")
        for i, (img_id, img_info) in enumerate(list(mapping.items())[:10]):
            description = img_info.get('enhanced_description', '无描述')
            image_path = img_info.get('image_path', '无路径')
            exists = os.path.exists(image_path) if image_path else False
        
            print(f"  {i+1}. {img_id}")
            print(f"     描述: {description[:100]}...")
            print(f"     路径: {image_path} {'✅' if exists else '❌'}")
            print()

    # 在适当的地方调用这个函数
    check_psychology_image_mapping()

    def test_conflict_resolution_images():
        """测试冲突解决相关的图片检索"""
        psychology_rag = MultiRAG(scene="psychology")
    
        test_queries = [
            "冲突解决",
            "朋友吵架", 
            "人际冲突",
            "矛盾解决",
            "沟通技巧",
            "情绪管理"
        ]
    
        for query in test_queries:
            print(f"\n🔍 测试查询: '{query}'")
            results = psychology_rag.retrieve(query, topk=10)
        
            image_results = [r for r in results if r.get('type') == 1]
            text_results = [r for r in results if r.get('type') == 0]
        
            print(f"  找到 {len(image_results)} 个图片, {len(text_results)} 个文本")
        
            for i, img in enumerate(image_results[:3]):
                print(f"    图片{i+1}: {img.get('document', '')[:80]}...")

    # 运行测试
    test_conflict_resolution_images()

#下面我模拟了剩下两个助手的类（方便在Intent_answer初始化时统一助手类名）
# 但他们的path未定，我先注释掉了初始化部分

class PaperAssistant(LLM_paper):
    def __init__(self, app_id=None):
        super().__init__(app_id or APP_ID)
        self.session_id = "paper_session"
        # 只需传递场景参数
        self.multirag = MultiRAG(scene="paper")
        print("论文助手 MultiRAG 系统初始化完成")

    def start_paper(self):
        """
        启动论文助手服务
        """
        return "论文助手启动成功"

    def retrieve_and_answer(self, query: str, top_k: int = 8):
        """
        智能检索并回答问题 - 论文助手专用

        Args:
            query (str): 用户问题
            top_k (int): 检索的片段数量

        Yields:
            str: 生成的文本段落
        """
        try:
            # 1. 使用MultiRAG检索相关片段
            print(f"论文助手: 正在检索与问题相关的top-{top_k}片段...")
            results = self.multirag.retrieve(query, topk=top_k)

            if not results:
                print("论文助手: 未找到相关片段，使用通用知识回答")
                yield from self.call_llm_stream(query, [])
                return

            # 2. 处理检索结果
            text_chunks = []
            image_info = []

            for result in results:
                result_type = result.get('type', 0)
                document = result.get('document', '')
                source = result.get('source', '')

                if result_type == 1:  # 图片类型
                    if source and source != "":
                        image_info.append({
                            'description': document,
                            'path': source,
                            'score': 1.0
                        })
                        text_chunks.append(f"[图表内容] {document} [图表地址: {source}]")
                    else:
                        text_chunks.append(f"[图表内容] {document}")
                else:
                    text_chunks.append(document)

            print(f"论文助手: 检索到 {len(text_chunks)} 个文本片段，{len(image_info)} 个图表")

            # 3. 构建增强的prompt
            enhanced_chunks = self._enhance_paper_chunks(text_chunks, image_info)

            # 4. 调用父类的流式生成方法
            yield from self.call_llm_stream(query, enhanced_chunks)

        except Exception as e:
            print(f"论文助手检索过程出错: {e}")
            import traceback
            traceback.print_exc()
            yield from self.call_llm_stream(query, [])

    def _enhance_paper_chunks(self, text_chunks, image_info):
        """
        根据图表信息增强论文文本片段
        """
        enhanced_chunks = text_chunks.copy()

        if image_info:
            image_instruction = "\n注意：回答中如需引用论文图表、数据可视化或实验图示，请直接使用图表地址，格式为：[具体路径]\n"
            enhanced_chunks.append(image_instruction)

            image_summary = "可用论文图表资源：\n"
            for i, img in enumerate(image_info[:3]):
                image_summary += f"{i + 1}. {img['description']} [地址: {img['path']}]\n"
            enhanced_chunks.append(image_summary)

        return enhanced_chunks

    def call_llm_stream(self, query, list):
        """
        重写父类的流式生成方法，添加论文助手专用的提示词增强
        """
        separator = "\n\n"
        # 使用父类的系统提示词，并添加论文专用增强
        system_prompt = self.get_stream_system_prompt()
        
        prompt = f"""{system_prompt}

请根据用户的问题和下面的背景知识进行回答。

用户问题: {query}

背景知识:
{separator.join(list)}

回答要求：
1. 用严谨、学术、专业的语言进行回答，保持论文写作风格。
2. 将完整的回答分成3到5段，段与段之间要在语义和逻辑上相互承接，段落之间必须用 `[NEW_PARAGRAPH]` 分隔。
3. 如果背景知识中包含论文图表、数据可视化或实验图示（标注为[图表内容]或[图表地址]），请在回答中适当引用。
4. 引用图表时，直接使用提供的图表地址，格式：[具体路径]，无需任何前缀或后缀。
5. 若用户问题与背景知识无关，则用通用学术知识解决问题。
6. 保持学术严谨性，同时要清晰易懂。

请开始你的回答：
"""

        # 使用父类的非流式调用逻辑
        full_response_text = ""
        try:
            response = Application.call(
                api_key=self.api_key,
                app_id=self.app_id,
                prompt=prompt,
                session_id=self.session_id,
                stream=False
            )
            if response.status_code == HTTPStatus.OK:
                request_id = response.request_id
                print(f"论文助手: 成功获取到回答，Request ID: {request_id}")
                full_response_text = response.output.text
            else:
                error_message = f'论文助手 API Error: {response.message}'
                print(error_message)
                yield error_message
                return

        except Exception as e:
            error_message = f"论文助手调用LLM时发生异常: {e}"
            print(error_message)
            yield error_message
            return

        # 根据分隔符切分段落并依次返回
        paragraphs = full_response_text.split('[NEW_PARAGRAPH]')
        for para in paragraphs:
            cleaned_para = para.strip()
            if cleaned_para:
                yield cleaned_para

class FitnessAssistant(LLM_fitness):
    def __init__(self, app_id=None):
        super().__init__(app_id or APP_ID)
        self.session_id = "fitness_session"
        # 只需传递场景参数
        self.multirag = MultiRAG(scene="fitness")
        print("健康饮食助手 MultiRAG 系统初始化完成")

    def start_fitness(self):
        """
        启动健康饮食助手服务
        """
        return "健康饮食助手启动成功"

    def retrieve_and_answer(self, query: str, top_k: int = 8):
        """
        智能检索并回答问题 - 健康饮食助手专用

        Args:
            query (str): 用户问题
            top_k (int): 检索的片段数量

        Yields:
            str: 生成的文本段落
        """
        try:
            # 1. 使用MultiRAG检索相关片段
            print(f"健康饮食助手: 正在检索与问题相关的top-{top_k}片段...")
            results = self.multirag.retrieve(query, topk=top_k)

            if not results:
                print("健康饮食助手: 未找到相关片段，使用通用知识回答")
                yield from self.call_llm_stream(query, [])
                return

            # 2. 处理检索结果
            text_chunks = []
            image_info = []

            for result in results:
                result_type = result.get('type', 0)
                document = result.get('document', '')
                source = result.get('source', '')

                if result_type == 1:  # 图片类型
                    if source and source != "":
                        image_info.append({
                            'description': document,
                            'path': source,
                            'score': 1.0
                        })
                        text_chunks.append(f"[动作图示] {document} [图示地址: {source}]")
                    else:
                        text_chunks.append(f"[动作图示] {document}")
                else:
                    text_chunks.append(document)

            print(f"健康饮食助手: 检索到 {len(text_chunks)} 个文本片段，{len(image_info)} 个动作图示")

            # 3. 构建增强的prompt
            enhanced_chunks = self._enhance_fitness_chunks(text_chunks, image_info)

            # 4. 调用父类的流式生成方法
            yield from self.call_llm_stream(query, enhanced_chunks)

        except Exception as e:
            print(f"健康饮食助手检索过程出错: {e}")
            import traceback
            traceback.print_exc()
            yield from self.call_llm_stream(query, [])

    def _enhance_fitness_chunks(self, text_chunks, image_info):
        """
        根据动作图示信息增强健身文本片段
        """
        enhanced_chunks = text_chunks.copy()

        if image_info:
            image_instruction = "\n注意：回答中如需引用健身动作图示、营养图表或解剖图示，请直接使用图示地址，格式为：[具体路径]\n"
            enhanced_chunks.append(image_instruction)

            image_summary = "可用健康饮食图示资源：\n"
            for i, img in enumerate(image_info[:3]):
                image_summary += f"{i + 1}. {img['description']} [地址: {img['path']}]\n"
            enhanced_chunks.append(image_summary)

        return enhanced_chunks

    def call_llm_stream(self, query, list):
        """
        重写父类的流式生成方法，添加健康饮食助手专用的提示词增强
        """
        separator = "\n\n"
        # 使用父类的系统提示词，并添加健身专用增强
        system_prompt = self.get_stream_system_prompt()
        
        prompt = f"""{system_prompt}

请根据用户的问题和下面的背景知识进行回答。

用户问题: {query}

背景知识:
{separator.join(list)}

回答要求：
1. 用鼓励、专业、实用的语言进行回答，保持健身教练风格。
2. 将完整的回答分成3到5段，段与段之间要在语义和逻辑上相互承接，段落之间必须用 `[NEW_PARAGRAPH]` 分隔。
3. 如果背景知识中包含健身动作图示、营养图表或解剖图示（标注为[动作图示]或[图示地址]），请在回答中适当引用。
4. 引用图示时，直接使用提供的图示地址，格式：[具体路径]，无需任何前缀或后缀。
5. 若用户问题与背景知识无关，则用通用健身营养知识解决问题。
6. 保持专业性，同时要鼓励和支持用户。

请开始你的回答：
"""

        # 使用父类的非流式调用逻辑
        full_response_text = ""
        try:
            response = Application.call(
                api_key=self.api_key,
                app_id=self.app_id,
                prompt=prompt,
                session_id=self.session_id,
                stream=False
            )
            if response.status_code == HTTPStatus.OK:
                request_id = response.request_id
                print(f"健康饮食助手: 成功获取到回答，Request ID: {request_id}")
                full_response_text = response.output.text
            else:
                error_message = f'健康饮食助手 API Error: {response.message}'
                print(error_message)
                yield error_message
                return

        except Exception as e:
            error_message = f"健康饮食助手调用LLM时发生异常: {e}"
            print(error_message)
            yield error_message
            return

        # 根据分隔符切分段落并依次返回
        paragraphs = full_response_text.split('[NEW_PARAGRAPH]')
        for para in paragraphs:
            cleaned_para = para.strip()
            if cleaned_para:
                yield cleaned_para