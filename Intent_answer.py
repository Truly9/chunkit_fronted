#=====================这是接受用户信息，获取回答的主函数===================
import os
callback_path = os.path.join('ClassAssistant', 'callback.py')
import sys
import json
from dotenv import load_dotenv
from IntentRecognition.Intent_by_Rag import RagQueryEnhancer
from ClassAssistant.callback import CampusAssistant, PsychologyAssistant, FitnessAssistant, PaperAssistant
import ClassAssistant.callback

print(f"callback.py 文件位置: {ClassAssistant.callback.__file__}")
# 加载 .env 文件
env_path = "Agent.env"
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
        """延迟初始化校园 Assistant - 增强调试"""
        if self.campus_assistant is None:
            try:
                self.campus_assistant = CampusAssistant(debug=self.debug)  
                result = self.campus_assistant.start_service()
            except Exception as e:
                print(f"❌ 校园 Assistant 初始化失败: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
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
    #             self.paper_assistant = PaperAssistant( use_fallback=True)
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
    #             self.fitness_assistant = FitnessAssistant(use_fallback=True)
    #             self.fitness_assistant.start_service()
    #             print("健身 Assistant 初始化成功")
    #         except Exception as e:
    #             print(f"健身 Assistant 初始化失败: {e}")
    #             return None
    #     return self.fitness_assistant
    def _format_response_with_avatar(self, response):
        """格式化响应，确保每个段落都有头像"""
        try:
            avatar = response.get('avatar', '')
            intent_name = response.get('intent', '')
            answer = response.get('answer', '')
            
            # 如果回答中包含 [NEW_PARAGRAPH]，则在每个段落前添加头像
            if '[NEW_PARAGRAPH]' in answer:
                paragraphs = answer.split('[NEW_PARAGRAPH]')
                # 第一个段落已经有头像，后续段落前都添加头像
                formatted_paragraphs = [paragraphs[0]]  # 第一个段落保持原样
                for para in paragraphs[1:]:
                    if para.strip():  # 非空段落
                        formatted_paragraphs.append(f"{avatar} {intent_name}: {para.strip()}")
                
                formatted_answer = '\n'.join(formatted_paragraphs)
            else:
                formatted_answer = answer
                
            response['formatted_answer'] = formatted_answer
            return response
            
        except Exception as e:
            print(f"格式化响应时出错: {e}")
            response['formatted_answer'] = response.get('answer', '')
            return response
        
    def _get_batch_answers_for_intents(self, enhancement_result: dict) -> list:
        """非流式处理 - 返回完整的回答和图片 - 增强错误处理"""
        all_responses = []
        original_query = enhancement_result.get("original_query")

        for item in enhancement_result["analysis_results"]:
            if "error" in item: 
                print(f"❌ 意图处理错误: {item['error']}")
                continue

            Rag_intent = item["intent"]
            avatar = self.intent_avatar_mapping.get(Rag_intent, self.intent_avatar_mapping["其他"])
            
            try:
                result_dict = None
                
                # 根据意图选择对应的 Assistant
                if Rag_intent == "校园知识问答助手":
                    campus_assistant = self.get_campus_assistant()
                    if campus_assistant:
                        try:
                            result_dict = campus_assistant.retrieve_and_answer(original_query, top_k=8, stream_mode=False)
                        except Exception as e:
                            print(f"❌ 校园助手处理失败: {e}")
                            import traceback
                            traceback.print_exc()
                            result_dict = {
                                "answer": f"校园助手处理时出现错误: {str(e)}",
                                "images": []
                            }
                    else:
                        print("❌ 校园助手初始化失败")
                        result_dict = {"answer": "抱歉，校园助手初始化失败。", "images": []}
                
                elif Rag_intent == "心理助手":
                    psychology_assistant = self.get_psychology_assistant()
                    if psychology_assistant:
                        try:
                            result_dict = psychology_assistant.retrieve_and_answer(original_query, top_k=8, stream_mode=False)
                        except Exception as e:
                            print(f"❌ 心理助手处理失败: {e}")
                            import traceback
                            traceback.print_exc()
                            result_dict = {
                                "answer": f"心理助手处理时出现错误: {str(e)}",
                                "images": []
                            }
                    else:
                        print("❌ 心理助手初始化失败")
                        result_dict = {"answer": "抱歉，心理学助手初始化失败。", "images": []}
                
                # 构建响应
                if result_dict:
                    answer = result_dict.get("answer", "")
                    
                    # 检查是否是API错误，如果是则提供更友好的提示
                    if "Arrearage" in answer or "欠费" in answer or "API" in answer:
                        answer = "当前AI服务暂时不可用，正在使用本地知识库为您提供信息。\n\n" + answer
                    # 处理分段显示
                    formatted_response = self._format_response_with_avatar({
                        "avatar": avatar,
                        "intent": Rag_intent,
                        "answer": answer
                    })
                    formatted_answer = formatted_response.get('formatted_answer', answer)
                    
                    # 处理图片信息
                    images = result_dict.get("images", [])
                    formatted_images = []
                    for img_info in images:
                        if img_info.get('status') == 'exists' and img_info.get('absolute_path'):
                            formatted_images.append({
                                'absolute_path': img_info['absolute_path'],
                                'filename': img_info.get('filename', os.path.basename(img_info['absolute_path'])),
                                'description': img_info.get('description', '')
                            })
                    
                    
                    all_responses.append({
                        "success": True, 
                        "intent": Rag_intent, 
                        "avatar": avatar, 
                        "answer": formatted_answer,
                        "images": formatted_images
                    })
                else:
                    print("调试: result_dict 为 None")
                    all_responses.append({
                        "success": False, 
                        "intent": Rag_intent, 
                        "avatar": avatar, 
                        "error": "未能获取到回答",
                        "images": []
                    })
                
            except Exception as e:
                error_msg = f"处理意图 '{Rag_intent}' 时发生错误: {str(e)}"
                print(f"{error_msg}")
                import traceback
                traceback.print_exc()
                all_responses.append({
                    "success": False, 
                    "intent": Rag_intent, 
                    "avatar": avatar, 
                    "error": error_msg,
                    "images": []
                })

        print(f" 调试: 最终返回 {len(all_responses)} 个响应")
        return all_responses
    
    def _stream_answers_for_intents(self, enhancement_result: dict):
        """流式处理 - 分段输出回答和图片 - 增强错误处理"""
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
                
                    # 根据意图选择对应的 Assistant
                    if Rag_intent == "校园知识问答助手":
                        campus_assistant = self.get_campus_assistant()
                        if campus_assistant:
                            try:
                                result_dict = campus_assistant.retrieve_and_answer(original_query, top_k=8, stream_mode=True)
                            except Exception as e:
                                print(f"校园助手流式处理失败: {e}")
                                # 生成错误响应
                                error_message = f"校园助手处理失败: {str(e)}"
                                result_dict = {"answer_generator": iter([error_message]), "images": []}
                        else:
                            result_dict = {"answer_generator": iter(["抱歉，校园助手初始化失败。"])}
                
                    elif Rag_intent == "心理助手":
                        psychology_assistant = self.get_psychology_assistant()
                        if psychology_assistant:
                            try:
                                result_dict = psychology_assistant.retrieve_and_answer(original_query, top_k=8, stream_mode=True)
                            except Exception as e:
                                print(f"心理助手流式处理失败: {e}")
                                error_message = f"心理助手处理失败: {str(e)}"
                                result_dict = {"answer_generator": iter([error_message]), "images": []}
                        else:
                            result_dict = {"answer_generator": iter(["抱歉，心理学助手初始化失败。"])}
                
                    # 处理流式输出
                    if result_dict and "answer_generator" in result_dict:
                        full_answer = ""
                        try:
                            for chunk in result_dict["answer_generator"]:
                                # 检查是否是错误信息
                                if "Arrearage" in chunk or "欠费" in chunk:
                                    chunk = "当前AI服务不可用，使用本地模式。\n" + chunk
                                
                                full_answer += chunk
                                yield {
                                    "type": "content",
                                    "intent": Rag_intent,
                                    "avatar": avatar,
                                    "delta": chunk
                                }
                        except Exception as e:
                            error_chunk = f"\n流式输出过程中发生错误: {str(e)}"
                            yield {
                                "type": "content", 
                                "intent": Rag_intent,
                                "avatar": avatar,
                                "delta": error_chunk
                            }

                        # 输出图片信息
                        images = result_dict.get("images", [])
                        formatted_images = []
                        for img_info in images:
                            if img_info.get('status') == 'exists' and img_info.get('source'):
                                formatted_images.append({
                                    'path': img_info['source'],
                                    'description': img_info.get('description', ''),
                                    'filename': img_info.get('filename', os.path.basename(img_info['source']))
                                })
                    
                        if formatted_images:
                            yield {
                                "type": "images",
                                "intent": Rag_intent,
                                "avatar": avatar,
                                "images": formatted_images
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

        # 默认使用非流式模式，更稳定
        stream_mode = False
        print("当前模式: 非流式模式 (更稳定)")

        while True:
            user_input = input("你：")

            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break

            if user_input.lower() == "batch":
                stream_mode = not stream_mode
                mode_name = "流式" if stream_mode else "非流式"
                print(f"模式已切换。当前: {mode_name}模式")
                continue

            # 添加处理提示
            print("处理中...", end="", flush=True)
            
            results = self.process_question_with_full_response(user_input, stream_mode=stream_mode)
            
            print("\r", end="")  # 清除"处理中"提示
        
            # 根据模式处理并打印结果
            if stream_mode:
                print("--- 流式回答 ---")
                # ... 流式处理代码保持不变
            else:
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
                        print(f"相关图片:")
                        for i, img_info in enumerate(images, 1):
                                abs_path = img_info.get('absolute_path', '')
                                if abs_path and os.path.exists(abs_path):
                                    print(f"    图片{i}: {os.path.basename(abs_path)}")
                                else:
                                    print(f"    图片{i}: 文件不存在或路径无效")
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