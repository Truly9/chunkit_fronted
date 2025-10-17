#=====================这是接受用户信息，获取回答的主函数===================
import os
import json
from dotenv import load_dotenv
from IntentRecognition.Intent_by_Rag import RagQueryEnhancer
from ClassAssistant.callback import CampusAssistant, PsychologyAssistant,FitnessAssistant, PaperAssistant

# 加载 .env 文件
env_path = "demo/back-end-python/chunkit_fronted/Agent.env"
load_dotenv(env_path)

# 验证环境变量是否设置
required_env_vars = [
    "BAILIAN_API_KEY",
    "APP_ID_PSYCHOLOGY",
    "APP_ID_CAMPUS",
    "APP_ID_FITNESS",
    "APP_ID_PAPER"
]

missing_vars = []
for var in required_env_vars:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    print(f"请在.env文件中设置以下环境变量: {', '.join(missing_vars)}")
    exit(1)

print("所有环境变量配置验证成功")
print(f"使用的智能体应用:")
print(f"   - 心理助手: {os.getenv('APP_ID_PSYCHOLOGY')}")
print(f"   - 健身助手: {os.getenv('APP_ID_FITNESS')}")
print(f"   - 校园助手: {os.getenv('APP_ID_CAMPUS')}")
print(f"   - 论文助手: {os.getenv('APP_ID_PAPER')}")
print()

enhancer = RagQueryEnhancer()

# ----修改部分：我在 InteractiveAgent 类中更新了所有助手调用，都支持了返回图片----#
# --我注释掉了论文和健康的所有操作--#
class InteractiveAgent:
    def __init__(self):
        try:
            print("意图分类器初始化成功")

            # 初始化意图识别增强器
            self.enhancer = RagQueryEnhancer()
            
            # 更新：移除原有的 RAG 智能体，全部使用新的 Assistant 类
            self.rag_agents = {}  # 保留为空，不再使用旧的 RAG 类
            
            # 更新：使用新的助手类
            self.campus_assistant = None      # 校园助手
            self.psychology_assistant = None  # 心理助手
            self.paper_assistant = None       # 论文助手
            self.fitness_assistant = None     # 健身助手
            
            # 意图到头像的映射关系保持不变
            self.intent_avatar_mapping = {
                "心理助手": "007-gin tonic.svg",
                "健身饮食助手": "014-mojito.svg", 
                "校园知识问答助手": "042-milkshake.svg",
                "论文助手": "044-whiskey sour.svg",
                "其他": "050-lemon juice.svg"
            }
            
            print("助手类架构初始化完成")

        except Exception as e:
            print(f"初始化失败: {e}")
            raise

    def process_question_with_full_response(self, user_input: str, stream_mode: bool = False):
        """处理用户问题并返回一个或多个完整的回答。这是主聊天流程调用的方法。"""
        try:
            # 1. 【第一步】进行意图识别和查询强化，这是所有后续操作的基础。
            enhancement_result = self.enhancer.enhance_query(user_input)

            # --- 【新增】可视化调试输出 ---
            if enhancement_result and enhancement_result.get("intent_distribution"):
                distribution = enhancement_result["intent_distribution"]
                total_docs = sum(distribution.values())

                # 构造调试信息字符串
                debug_parts = []
                for intent, count in distribution.items():
                    confidence = f"({count}/{total_docs})" if total_docs > 0 else ""
                    debug_parts.append(f"{intent} 有 {count} 份 {confidence}")

                print(f"🔍 [调试信息] 检索到的意图分布: {', '.join(debug_parts)}")
            # --- 可视化结束 ---

            if not enhancement_result or not enhancement_result.get("analysis_results"):
                # 如果没结果，根据模式返回错误信息
                if stream_mode: 
                    return self._stream_error("抱歉，未能识别出您问题的意图。")
                return [{"success": False, "message": "未能识别出意图"}]

            # 2. 【第二步】根据模式，调用对应的执行器
            if stream_mode:
                # 流式模式下，将分析结果交给专门的流式生成器处理
                return self._stream_answers_for_intents(enhancement_result)
            else:
                # 非流式模式下，将分析结果交给专门的批量处理器处理
                return self._get_batch_answers_for_intents(enhancement_result)

        except Exception as e:
            # 统一的顶层异常处理
            if stream_mode: 
                return self._stream_error(f"处理过程中发生严重错误: {str(e)}")
            return [{"success": False, "message": f"处理过程中发生严重错误: {str(e)}"}]


    def get_campus_assistant(self):
        """延迟初始化校园 Assistant"""
        if self.campus_assistant is None:
            print("🔧 正在初始化校园 Assistant...")
            try:
                self.campus_assistant = CampusAssistant()
                # 调用校园助手的启动方法
                self.campus_assistant.start_LLM()
                print("校园 Assistant 初始化成功")
            except Exception as e:
                print(f"校园 Assistant 初始化失败: {e}")
                return None
        return self.campus_assistant

    def get_psychology_assistant(self):
        """延迟初始化心理 Assistant"""
        if self.psychology_assistant is None:
            print("🔧 正在初始化心理学 Assistant...")
            try:
                self.psychology_assistant = PsychologyAssistant()
                # 调用心理学助手的启动方法
                self.psychology_assistant.start_psychology()
                print("心理学 Assistant 初始化成功")
            except Exception as e:
                print(f"心理学 Assistant 初始化失败: {e}")
                return None
        return self.psychology_assistant


    # def get_paper_assistant(self):
    #     """延迟初始化论文 Assistant"""
    #     if self.paper_assistant is None:
    #         print("🔧 正在初始化论文 Assistant...")
    #         try:
    #             self.paper_assistant = PaperAssistant()
    #             # 调用论文助手的启动方法
    #             self.paper_assistant.start_paper()
    #             print("论文 Assistant 初始化成功")
    #         except Exception as e:
    #             print(f"论文 Assistant 初始化失败: {e}")
    #             return None
    #     return self.paper_assistant


    # def get_fitness_assistant(self):
    #     """延迟初始化健身 Assistant"""
    #     if self.fitness_assistant is None:
    #         print("🔧 正在初始化健身 Assistant...")
    #         try:
    #             self.fitness_assistant = FitnessAssistant()
    #             # 调用健身助手的启动方法
    #             self.fitness_assistant.start_fitness()
    #             print("健身 Assistant 初始化成功")
    #         except Exception as e:
    #             print(f"健身 Assistant 初始化失败: {e}")
    #             return None
    #     return self.fitness_assistant


    # 在非流式处理中更新所有助手调用
    def _get_batch_answers_for_intents(self, enhancement_result: dict) -> list:
        all_responses = []
        original_query = enhancement_result.get("original_query")
        
        for item in enhancement_result["analysis_results"]:
            if "error" in item: 
                continue

            Rag_intent = item["intent"]
            rewritten_query = item["rewritten_query"]
            avatar = self.intent_avatar_mapping.get(Rag_intent, self.intent_avatar_mapping["其他"])

            try:
                # 根据意图选择对应的 Assistant
                if Rag_intent == "校园知识问答助手":
                    campus_assistant = self.get_campus_assistant()
                    if campus_assistant:
                        # 使用 retrieve_and_answer 方法获取文本
                        answer_generator = campus_assistant.retrieve_and_answer(original_query, top_k=8)
                        answer = "".join(answer_generator)
                        # 校园助手也需要返回图片
                        images = self._get_campus_images_with_mapping(original_query, campus_assistant)
                    else:
                        answer = "抱歉，校园助手初始化失败。"
                        images = []
                
                elif Rag_intent == "心理助手":
                    psychology_assistant = self.get_psychology_assistant()
                    if psychology_assistant:
                        # 使用 retrieve_with_images 方法
                        result = psychology_assistant.retrieve_with_images(original_query, top_k=8)
                        answer = result.get("answer", "")
                        images = result.get("images", [])
                    else:
                        answer = "抱歉，心理学助手初始化失败。"
                        images = []
                
                elif Rag_intent == "论文助手":
                    paper_assistant = self.get_paper_assistant()
                    if paper_assistant:
                        # 使用 retrieve_and_answer 方法获取文本
                        answer_generator = paper_assistant.retrieve_and_answer(original_query, top_k=8)
                        answer = "".join(answer_generator)
                        # 论文助手也需要返回图片
                        images = self._get_paper_images(original_query, paper_assistant)
                    else:
                        answer = "抱歉，论文助手初始化失败。"
                        images = []
                
                elif Rag_intent == "健身饮食助手":
                    fitness_assistant = self.get_fitness_assistant()
                    if fitness_assistant:
                        # 使用 retrieve_and_answer 方法获取文本
                        answer_generator = fitness_assistant.retrieve_and_answer(original_query, top_k=8)
                        answer = "".join(answer_generator)
                        # 健身助手也需要返回图片
                        images = self._get_fitness_images(original_query, fitness_assistant)
                    else:
                        answer = "抱歉，健身助手初始化失败。"
                        images = []
                
                else:
                    # 对于未知意图，返回错误信息
                    answer = "抱歉，暂不支持此意图。"
                    images = []

                # 如果有图片，在答案中添加图片提示
                if images:
                    answer += "\n\n📷 相关图片:\n"
                    for i, img_info in enumerate(images, 1):
                        img_path = img_info.get('source', '') or img_info.get('path', '')
                        if img_path and os.path.exists(img_path):
                            answer += f"图片{i}: {img_path}\n"
                        else:
                            answer += f"图片{i}: 文件不存在\n"

                all_responses.append({
                    "success": True, 
                    "intent": Rag_intent, 
                    "avatar": avatar, 
                    "answer": answer,
                    "images": images  # 添加图片信息
                })
            except Exception as e:
                all_responses.append({
                    "success": False, 
                    "intent": Rag_intent, 
                    "avatar": avatar, 
                    "error": str(e),
                    "images": []
                })
        
        return all_responses

    def _get_campus_images_with_mapping(self, query: str, campus_assistant) -> list:
        """【修复】获取校园助手的图片信息 - 使用映射文件"""
        try:
            # 1. 使用 MultiRAG 检索
            results = campus_assistant.multirag.retrieve(query, topk=8)
            images = []
            
            # 2. 获取映射文件路径
            mapping_file = getattr(campus_assistant.multirag, 'image_mapping_file', '')
            if not mapping_file or not os.path.exists(mapping_file):
                print(f"❌ 校园助手映射文件不存在: {mapping_file}")
                return []
            
            # 3. 加载映射文件
            with open(mapping_file, 'r', encoding='utf-8') as f:
                image_mapping = json.load(f)
            
            # 4. 处理检索结果
            for result in results:
                result_type = result.get('type', 0)
                if result_type == 1:  # 图片类型
                    # 获取图片ID
                    content = result.get('content', '')
                    if content.startswith('image_'):
                        # 提取图片ID（格式：image_psychology_xxxx）
                        image_id = content.split(':', 1)[0].strip()
                        
                        # 从映射文件中查找图片信息
                        if image_id in image_mapping:
                            img_info = image_mapping[image_id]
                            img_path = img_info.get('image_path', '')
                            
                            # 验证图片文件是否存在
                            if img_path and os.path.exists(img_path):
                                images.append({
                                    'source': img_path,
                                    'description': img_info.get('enhanced_description', '')[:100] + '...',
                                    'score': result.get('score', 0)
                                })
                            else:
                                print(f"⚠️ 图片文件不存在: {img_path}")
            
            return images[:3]  # 返回前3个图片
        except Exception as e:
            print(f"❌ 获取校园图片时出错: {e}")
            return []

    def _get_campus_images(self, query: str, campus_assistant) -> list:
        """【保留原有方法】获取校园助手的图片信息"""
        try:
            # 直接使用 MultiRAG 检索图片
            results = campus_assistant.multirag.retrieve(query, topk=8)
            images = []
            
            for result in results:
                result_type = result.get('type', 0)
                if result_type == 1:  # 图片类型
                    source = result.get('source', '')
                    document = result.get('document', '')
                    if source and source != "":
                        images.append({
                            'source': source,
                            'description': document[:100] + '...' if len(document) > 100 else document,
                            'score': result.get('score', 0)
                        })
            
            return images[:3]  # 返回前3个图片
        except Exception as e:
            print(f"获取校园图片时出错: {e}")
            return []


    # def _get_paper_images(self, query: str, paper_assistant) -> list:
    #     """获取论文助手的图片信息"""
    #     try:
    #         # 直接使用 MultiRAG 检索图片
    #         results = paper_assistant.multirag.retrieve(query, topk=8)
    #         images = []
            
    #         for result in results:
    #             result_type = result.get('type', 0)
    #             if result_type == 1:  # 图片类型
    #                 source = result.get('source', '')
    #                 document = result.get('document', '')
    #                 if source and source != "":
    #                     images.append({
    #                         'source': source,
    #                         'description': document[:100] + '...' if len(document) > 100 else document,
    #                         'score': result.get('score', 0)
    #                     })
            
    #         return images[:3]  # 返回前3个图片
    #     except Exception as e:
    #         print(f"获取论文图片时出错: {e}")
    #         return []

    # def _get_fitness_images(self, query: str, fitness_assistant) -> list:
    #     """获取健身助手的图片信息"""
    #     try:
    #         # 直接使用 MultiRAG 检索图片
    #         results = fitness_assistant.multirag.retrieve(query, topk=8)
    #         images = []
            
    #         for result in results:
    #             result_type = result.get('type', 0)
    #             if result_type == 1:  # 图片类型
    #                 source = result.get('source', '')
    #                 document = result.get('document', '')
    #                 if source and source != "":
    #                     images.append({
    #                         'source': source,
    #                         'description': document[:100] + '...' if len(document) > 100 else document,
    #                         'score': result.get('score', 0)
    #                     })
            
    #         return images[:3]  # 返回前3个图片
    #     except Exception as e:
    #         print(f"获取健身图片时出错: {e}")
    #         return []


    # 在流式处理中更新所有助手调用
    def _stream_answers_for_intents(self, enhancement_result: dict):
        try:
            original_query = enhancement_result.get("original_query")
            if not original_query:
                yield from self._stream_error("未能获取到原始用户问题。")
                return

            for item in enhancement_result["analysis_results"]:
                if "error" in item:
                    yield {"type": "error", "intent": item.get("intent"), "message": item["error"]}
                    continue

                Rag_intent = item["intent"]
                avatar = self.intent_avatar_mapping.get(Rag_intent, self.intent_avatar_mapping["其他"])

                paragraph_generator = None
                images = []  # 存储图片信息

                try:
                    if Rag_intent == "校园知识问答助手":
                        campus_assistant = self.get_campus_assistant()
                        if campus_assistant:
                            paragraph_generator = campus_assistant.retrieve_and_answer(original_query, top_k=8)
                            # 【修复】使用新的图片获取方法
                            images = self._get_campus_images_with_mapping(original_query, campus_assistant)
                        else:
                            paragraph_generator = iter(["抱歉，校园助手初始化失败。"])
                    
                    elif Rag_intent == "心理助手":
                        psychology_assistant = self.get_psychology_assistant()
                        if psychology_assistant:
                            # 使用 retrieve_with_images 方法获取结果
                            result = psychology_assistant.retrieve_with_images(original_query, top_k=8)
                            answer_text = result.get("answer", "")
                            images = result.get("images", [])
                            paragraph_generator = iter([answer_text])
                        else:
                            paragraph_generator = iter(["抱歉，心理学助手初始化失败。"])
                    
                    # elif Rag_intent == "论文助手":
                    #     paper_assistant = self.get_paper_assistant()
                    #     if paper_assistant:
                    #         paragraph_generator = paper_assistant.retrieve_and_answer(original_query, top_k=8)
                    #         # 获取论文图片
                    #         images = self._get_paper_images(original_query, paper_assistant)
                    #     else:
                    #         paragraph_generator = iter(["抱歉，论文助手初始化失败。"])
                    
                    # elif Rag_intent == "健身饮食助手":
                    #     fitness_assistant = self.get_fitness_assistant()
                    #     if fitness_assistant:
                    #         paragraph_generator = fitness_assistant.retrieve_and_answer(original_query, top_k=8)
                    #         # 获取健身图片
                    #         images = self._get_fitness_images(original_query, fitness_assistant)
                    #     else:
                    #         paragraph_generator = iter(["抱歉，健身助手初始化失败。"])
                    
                    else:
                        paragraph_generator = iter(["抱歉，暂不支持此意图。"])

                    # 统一处理所有段落流
                    if paragraph_generator:
                        for paragraph in paragraph_generator:
                            yield {
                                "type": "content",
                                "intent": Rag_intent,
                                "avatar": avatar,
                                "delta": paragraph
                            }

                    # 如果有图片，发送图片信息
                    if images:
                        yield {
                            "type": "images",
                            "intent": Rag_intent,
                            "avatar": avatar,
                            "images": images
                        }

                except Exception as e:
                    yield {
                        "type": "error",
                        "intent": Rag_intent,
                        "avatar": avatar,
                        "message": f"处理时发生错误: {str(e)}"
                    }

                yield {"type": "break", "message": f"意图 {Rag_intent} 回答结束"}

            yield {"type": "finished", "finished": True}
        except Exception as e:
            yield from self._stream_error(f"流式处理时发生严重错误: {str(e)}")

    def _stream_error(self, message: str):
        """【辅助函数】用于在流式模式下返回一个标准的错误信息。"""
        yield {"type": "error", "message": message}
        yield {"type": "finished", "finished": True}

    def predict_intent_only(self, user_input):
        """
        进行意图识别，返回一个或多个意图及其对应的头像。

        Args:
            user_input (str): 用户输入的问题

        Returns:
            dict: 一个包含处理结果的字典。
                  - success (bool): 处理是否成功。
                  - results (list): 一个包含所有识别出的意图信息的列表。
                                    每个元素是一个字典，如:
                                    {"intent": "心理助手", "avatar": "🧠"}
                  - message (str): 描述信息。
        """
        try:
            # 进行意图识别
            enhancement_result = enhancer.enhance_query(user_input)
            # 检查是否有有效的分析结果
            if not enhancement_result or not enhancement_result.get("analysis_results"):
                return {
                    "success": False,
                    "results": [],
                    "message": "未能识别出任何意图"
                }

            # 2. 【关键】创建一个空列表，用于收集所有结果
            identified_intents = []

            # 3.遍历所有分析出的意图
            for item in enhancement_result["analysis_results"]:
                if "error" in item:
                    print(f"处理意图 '{item.get('intent', '未知')}' 时出错: {item['error']}")
                    continue  # 跳过这个出错的结果，继续下一个
                # 在循环内部获取每个意图
                Rag_intent = item["intent"]

                # 获取对应的头像
                avatar = self.intent_avatar_mapping.get(Rag_intent, self.intent_avatar_mapping["其他"])

                # 保存结果
                identified_intents.append({
                    "intent": Rag_intent,
                    "avatar": avatar
                })

            # 4.返回包含结果的列表
            if not identified_intents:
                return {
                    "success": False,
                    "results": [],
                    "message": "未能识别出任何有效意图"
                }

            return {
                "success": True,
                "results": identified_intents,  # 返回包含一个或多个结果的列表
                "message": f"成功识别出 {len(identified_intents)} 个意图"
            }

        except Exception as e:
            # 保持异常处理不变
            return {
                "success": False,
                "results": [],
                "error": str(e),
                "message": "意图识别过程中发生未知错误"
            }

    def chat(self):
        print("=== 欢迎使用智能助手系统 ===")
        print("本系统使用本地RAG检索增强 + 远程智能体架构")
        print("支持交叉编码器精确检索和流式回答")
        print("输入你的问题（输入 'exit' 退出，'batch' 切换非流式模式）：\n")

        stream_mode = True

        while True:
            user_input = input("你：")

            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break

            if user_input.lower() == "batch":
                stream_mode = not stream_mode
                print(f"模式已切换。当前流式输出: {'开启' if stream_mode else '关闭'}")
                continue

            results = self.process_question_with_full_response(user_input, stream_mode=stream_mode)
            # 根据模式处理并打印结果
            if stream_mode:
                # 处理流式生成器
                current_intent = "未知意图"
                print("--- 流式回答 (一段一段) ---")
                try:
                    for chunk in results:
                        # 处理 content 类型的包
                        if chunk.get('type') == 'content':
                            avatar = chunk.get('avatar', '🤖')
                            paragraph = chunk.get('delta', '')
                            print(f"头像: {avatar} | 回答段落: {paragraph}")

                        # 处理 images 类型的包
                        elif chunk.get('type') == 'images':
                            avatar = chunk.get('avatar', '🤖')
                            images = chunk.get('images', [])
                            print(f"头像: {avatar} | 相关图片:")
                            for i, img_info in enumerate(images, 1):
                                img_path = img_info.get('source', '')
                                if os.path.exists(img_path):
                                    print(f"  图片{i}: {img_path}")
                                else:
                                    print(f"  图片{i}: 文件不存在 - {img_path}")

                        elif chunk.get('type') == 'break':
                            print("--- (一个意图回答结束) ---\n")

                        elif chunk.get('type') == 'error':
                            print(f"处理时发生错误: {chunk.get('message')}")

                except Exception as e:
                    print(f"\n处理流式响应时发生错误: {e}")
                print("\n------------------\n")

            else:
                # 处理非流式（批量）结果
                print("--- 回答 ---")
                if not results:
                    print("抱歉，未能生成回答。")

                for response in results:
                    if response.get("success"):
                        intent = response.get('intent', '未知意图')
                        answer = response.get('answer', '（无回答）')
                        images = response.get('images', [])
                        
                        print(f"🤖 {intent} 回答：{answer}")
                        
                        # 显示图片信息
                        if images:
                            print(f"📷 {intent} 相关图片:")
                            for i, img_info in enumerate(images, 1):
                                img_path = img_info.get('source', '')
                                if os.path.exists(img_path):
                                    print(f"  图片{i}: {img_path}")
                                else:
                                    print(f"  图片{i}: 文件不存在 - {img_path}")
                        print()
                    else:
                        intent = response.get('intent', '未知意图')
                        error_msg = response.get('error', '未知错误')
                        print(f"处理意图 '{intent}' 时出错: {error_msg}\n")
                print("------------\n")


if __name__ == "__main__":
    try:
        agent = InteractiveAgent()
        agent.chat() 
        
    except KeyboardInterrupt:
        print("\n 程序被用户中断，再见！")
    except Exception as e:
        print(f"程序运行失败: {e}")