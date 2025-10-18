#=====================这是接受用户信息，获取回答的主函数===================
import os
import json
from dotenv import load_dotenv
from IntentRecognition.Intent_by_Rag import RagQueryEnhancer
from ClassAssistant.callback import CampusAssistant, PsychologyAssistant, FitnessAssistant, PaperAssistant

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

class InteractiveAgent:
    def __init__(self, debug=False):
        try:
            print("意图分类器初始化成功")
            self.debug = debug

            # 初始化意图识别增强器
            self.enhancer = RagQueryEnhancer()
            
            # 助手实例
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
        """处理用户问题并返回一个或多个完整的回答"""
        try:
            # 1. 进行意图识别和查询强化
            enhancement_result = self.enhancer.enhance_query(user_input)

            # 可视化调试输出
            if enhancement_result and enhancement_result.get("intent_distribution"):
                distribution = enhancement_result["intent_distribution"]
                total_docs = sum(distribution.values())

                debug_parts = []
                for intent, count in distribution.items():
                    confidence = f"({count}/{total_docs})" if total_docs > 0 else ""
                    debug_parts.append(f"{intent} 有 {count} 份 {confidence}")

                print(f"[调试信息] 检索到的意图分布: {', '.join(debug_parts)}")

            if not enhancement_result or not enhancement_result.get("analysis_results"):
                if stream_mode: 
                    return self._stream_error("抱歉，未能识别出您问题的意图。")
                return [{"success": False, "message": "未能识别出意图"}]

            # 2. 根据模式调用对应的执行器
            if stream_mode:
                return self._stream_answers_for_intents(enhancement_result)
            else:
                return self._get_batch_answers_for_intents(enhancement_result)

        except Exception as e:
            if stream_mode: 
                return self._stream_error(f"处理过程中发生严重错误: {str(e)}")
            return [{"success": False, "message": f"处理过程中发生严重错误: {str(e)}"}]

    def get_campus_assistant(self):
        """延迟初始化校园 Assistant"""
        if self.campus_assistant is None:
            print("正在初始化校园 Assistant...")
            try:
                self.campus_assistant = CampusAssistant(debug=self.debug)
                self.campus_assistant.start_service()
                print("校园 Assistant 初始化成功")
            except Exception as e:
                print(f"校园 Assistant 初始化失败: {e}")
                return None
        return self.campus_assistant

    def get_psychology_assistant(self):
        """延迟初始化心理 Assistant"""
        if self.psychology_assistant is None:
            print("正在初始化心理学 Assistant...")
            try:
                self.psychology_assistant = PsychologyAssistant(debug=self.debug)
                self.psychology_assistant.start_service()
                print("心理学 Assistant 初始化成功")
            except Exception as e:
                print(f"心理学 Assistant 初始化失败: {e}")
                return None
        return self.psychology_assistant

    # def get_paper_assistant(self):
    #     """延迟初始化论文 Assistant"""
    #     if self.paper_assistant is None:
    #         print("正在初始化论文 Assistant...")
    #         try:
    #             self.paper_assistant = PaperAssistant()
    #             self.paper_assistant.start_service()
    #             print("论文 Assistant 初始化成功")
    #         except Exception as e:
    #             print(f"论文 Assistant 初始化失败: {e}")
    #             return None
    #     return self.paper_assistant

    # def get_fitness_assistant(self):
    #     """延迟初始化健身 Assistant"""
    #     if self.fitness_assistant is None:
    #         print("正在初始化健身 Assistant...")
    #         try:
    #             self.fitness_assistant = FitnessAssistant()
    #             self.fitness_assistant.start_service()
    #             print("健身 Assistant 初始化成功")
    #         except Exception as e:
    #             print(f"健身 Assistant 初始化失败: {e}")
    #             return None
    #     return self.fitness_assistant

    def _get_batch_answers_for_intents(self, enhancement_result: dict) -> list:
        """非流式处理 - 返回完整的回答和图片"""
        all_responses = []
        original_query = enhancement_result.get("original_query")
        
        for item in enhancement_result["analysis_results"]:
            if "error" in item: 
                continue

            Rag_intent = item["intent"]
            avatar = self.intent_avatar_mapping.get(Rag_intent, self.intent_avatar_mapping["其他"])

            try:
                result_dict = None
                
                # 根据意图选择对应的 Assistant
                if Rag_intent == "校园知识问答助手":
                    campus_assistant = self.get_campus_assistant()
                    if campus_assistant:
                        result_dict = campus_assistant.retrieve_and_answer(original_query, top_k=8, stream_mode=False)
                    else:
                        result_dict = {"answer": "抱歉，校园助手初始化失败。", "images": []}
                
                elif Rag_intent == "心理助手":
                    psychology_assistant = self.get_psychology_assistant()
                    if psychology_assistant:
                        result_dict = psychology_assistant.retrieve_and_answer(original_query, top_k=8, stream_mode=False)
                    else:
                        result_dict = {"answer": "抱歉，心理学助手初始化失败。", "images": []}
                
                # elif Rag_intent == "论文助手":
                #     paper_assistant = self.get_paper_assistant()
                #     if paper_assistant:
                #         result_dict = paper_assistant.retrieve_and_answer(original_query, top_k=8, stream_mode=False)
                #     else:
                #         result_dict = {"answer": "抱歉，论文助手初始化失败。", "images": []}
                
                # elif Rag_intent == "健身饮食助手":
                #     fitness_assistant = self.get_fitness_assistant()
                #     if fitness_assistant:
                #         result_dict = fitness_assistant.retrieve_and_answer(original_query, top_k=8, stream_mode=False)
                #     else:
                #         result_dict = {"answer": "抱歉，健康饮食助手初始化失败。", "images": []}
                
                else:
                    result_dict = {"answer": "抱歉，暂不支持此意图。", "images": []}

                # 构建响应
                if result_dict:
                    answer = result_dict.get("answer", "")
                    # 处理分段显示
                    paragraphs = answer.split('[NEW_PARAGRAPH]')
                    formatted_answer = '\n\n'.join([p.strip() for p in paragraphs if p.strip()])
                    
                    all_responses.append({
                        "success": True, 
                        "intent": Rag_intent, 
                        "avatar": avatar, 
                        "answer": formatted_answer,
                        "images": result_dict.get("images", [])
                    })
                else:
                    all_responses.append({
                        "success": False, 
                        "intent": Rag_intent, 
                        "avatar": avatar, 
                        "error": "未能获取到回答",
                        "images": []
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

    def _stream_answers_for_intents(self, enhancement_result: dict):
        """流式处理 - 分段输出回答和图片"""
        try:
            original_query = enhancement_result.get("original_query")
            if not original_query:
                yield from self._stream_error("未能获取到用户问题。")
                return

            for item in enhancement_result["analysis_results"]:
                if "error" in item:
                    yield {"type": "error", "intent": item.get("intent"), "message": item["error"]}
                    continue

                Rag_intent = item["intent"]
                avatar = self.intent_avatar_mapping.get(Rag_intent, self.intent_avatar_mapping["其他"])

                try:
                    result_dict = None
                    
                    # 根据意图选择对应的 Assistant，传入 stream_mode=True
                    if Rag_intent == "校园知识问答助手":
                        campus_assistant = self.get_campus_assistant()
                        if campus_assistant:
                            result_dict = campus_assistant.retrieve_and_answer(original_query, top_k=8, stream_mode=True)
                        else:
                            result_dict = {"answer_generator": iter(["抱歉，校园助手初始化失败。"])}
                    
                    elif Rag_intent == "心理助手":
                        psychology_assistant = self.get_psychology_assistant()
                        if psychology_assistant:
                            result_dict = psychology_assistant.retrieve_and_answer(original_query, top_k=8, stream_mode=True)
                        else:
                            result_dict = {"answer_generator": iter(["抱歉，心理学助手初始化失败。"])}
                    
                    # elif Rag_intent == "论文助手":
                    #     paper_assistant = self.get_paper_assistant()
                    #     if paper_assistant:
                    #         result_dict = paper_assistant.retrieve_and_answer(original_query, top_k=8, stream_mode=True)
                    #     else:
                    #         result_dict = {"answer_generator": iter(["抱歉，论文助手初始化失败。"])}
                    
                    # elif Rag_intent == "健身饮食助手":
                    #     fitness_assistant = self.get_fitness_assistant()
                    #     if fitness_assistant:
                    #         result_dict = fitness_assistant.retrieve_and_answer(original_query, top_k=8, stream_mode=True)
                    #     else:
                    #         result_dict = {"answer_generator": iter(["抱歉，健康饮食助手初始化失败。"])}
                    
                    else:
                        result_dict = {"answer_generator": iter(["抱歉，暂不支持此意图。"])}

                    # 处理流式输出
                    if result_dict and "answer_generator" in result_dict:
                        full_answer = ""
                        for chunk in result_dict["answer_generator"]:
                            full_answer += chunk
                            yield {
                                "type": "content",
                                "intent": Rag_intent,
                                "avatar": avatar,
                                "delta": chunk
                            }

                        # 输出图片信息
                        images = result_dict.get("images", [])
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
        """用于在流式模式下返回一个标准的错误信息"""
        yield {"type": "error", "message": message}
        yield {"type": "finished", "finished": True}

    def predict_intent_only(self, user_input):
        """
        进行意图识别，返回一个或多个意图及其对应的头像
        """
        try:
            enhancement_result = self.enhancer.enhance_query(user_input)
            
            if not enhancement_result or not enhancement_result.get("analysis_results"):
                return {
                    "success": False,
                    "results": [],
                    "message": "未能识别出任何意图"
                }

            identified_intents = []

            for item in enhancement_result["analysis_results"]:
                if "error" in item:
                    print(f"处理意图 '{item.get('intent', '未知')}' 时出错: {item['error']}")
                    continue
                
                Rag_intent = item["intent"]
                avatar = self.intent_avatar_mapping.get(Rag_intent, self.intent_avatar_mapping["其他"])

                identified_intents.append({
                    "intent": Rag_intent,
                    "avatar": avatar
                })

            if not identified_intents:
                return {
                    "success": False,
                    "results": [],
                    "message": "未能识别出任何有效意图"
                }

            return {
                "success": True,
                "results": identified_intents,
                "message": f"成功识别出 {len(identified_intents)} 个意图"
            }

        except Exception as e:
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

        # 显示助手标识
        print("可用助手:")
        for intent, avatar in self.intent_avatar_mapping.items():
            print(f"  {avatar} {intent}")
        print()

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
                print("--- 流式回答 (一段一段) ---")
                try:
                    for chunk in results:
                        if chunk.get('type') == 'content':
                            intent = chunk.get('intent', '未知意图')
                            avatar = chunk.get('avatar', '')
                            delta = chunk.get('delta', '')
                        
                            # 处理段落分隔符
                            processed_delta = delta.replace('[NEW_PARAGRAPH]', '\n\n  ')
                        
                            # 只在意图开始时显示头像
                            if not hasattr(self, 'current_intent') or intent != getattr(self, 'current_intent', None):
                                print(f"{avatar} {intent}: {processed_delta}", end="", flush=True)
                                self.current_intent = intent
                            else:
                                print(processed_delta, end="", flush=True)

                        elif chunk.get('type') == 'images':
                            avatar = chunk.get('avatar', '')
                            images = chunk.get('images', [])
                            print(f"\n{avatar} 相关图片:")
                            for i, img_info in enumerate(images, 1):
                                img_path = img_info.get('source', '') or img_info.get('path', '')
                                if img_path and os.path.exists(img_path):
                                    print(f"图片{i}: {os.path.basename(img_path)}")
                                else:
                                    print(f"图片{i}: 文件不存在")

                        elif chunk.get('type') == 'break':
                            print("\n--- (一个意图回答结束) ---\n")
                            # 重置当前意图
                            if hasattr(self, 'current_intent'):
                                del self.current_intent

                        elif chunk.get('type') == 'error':
                            print(f"\n错误: {chunk.get('message')}")

                        elif chunk.get('type') == 'finished':
                            print("\n所有回答完成\n")
    
                except Exception as e:
                    print(f"\n处理流式响应时发生错误: {e}")
                    import traceback
                    traceback.print_exc()
                print("\n------------------\n")

            else:
                # 非流式模式保持不变
                print("--- 回答 ---")
                if not results:
                    print("抱歉，未能生成回答。")

                for response in results:
                    if response.get("success"):
                        intent = response.get('intent', '未知意图')
                        avatar = response.get('avatar', '')
                        answer = response.get('answer', '（无回答）')
                        images = response.get('images', [])
                    
                        print(f"{avatar} {intent}: {answer}")
                    
                        if images:
                           print(f"  📷 相关图片:")
                           for i, img_info in enumerate(images, 1):
                                img_path = img_info.get('source', '') or img_info.get('path', '')
                                if img_path and os.path.exists(img_path):
                                    print(f"    图片{i}: {os.path.basename(img_path)}")
                                else:
                                    print(f"图片{i}: 文件不存在")
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
        print("\n程序被用户中断，再见！")
    except Exception as e:
        print(f"程序运行失败: {e}")