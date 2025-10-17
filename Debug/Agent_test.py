import os
from dotenv import load_dotenv

# --- 关键修正 ---

from IntentRecognition.Intent_by_Rag import RagQueryEnhancer
from RAGlibrary import RAG_psychology, RAG_fitness, RAG_compus, RAG_paper
from callback import LLM_model

# 加载 .env 文件
load_dotenv("Agent.env")

# 验证环境变量
required_env_vars = [
    "BAILIAN_API_KEY", "APP_ID_PSYCHOLOGY", "APP_ID_CAMPUS",
    "APP_ID_FITNESS", "APP_ID_PAPER"
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"警告: 请在.env文件中或环境中设置以下变量: {', '.join(missing_vars)}")
else:
    print("所有必需的环境变量均已配置。")


enhancer = RagQueryEnhancer()

class InteractiveAgent:
    def __init__(self):
        try:
            print("意图分类器初始化成功")

            self.rag_agents = {}
            self.agent_classes = {
                "心理助手": RAG_psychology,
                "健身饮食助手": RAG_fitness,
                "校园知识问答助手": RAG_compus,
                "论文助手": RAG_paper
            }

            self.llm = LLM_model()
            self.llm.start_LLM()


            self.intent_avatar_mapping = {
                "心理助手": "007-gin tonic.svg",
                "健身饮食助手": "014-mojito.svg",
                "校园知识问答助手": "042-milkshake.svg",
                "论文助手": "044-whiskey sour.svg",
                "其他": "050-lemon juice.svg"
            }
            print("RAG 智能体类加载成功")
        except Exception as e:
            print(f"初始化失败: {e}")
            raise

    def get_rag_agent(self, intent):
        if intent not in self.rag_agents:
            if intent in self.agent_classes:
                print(f"🔧 正在初始化 {intent} RAG智能体...")
                try:
                    self.rag_agents[intent] = self.agent_classes[intent]()
                    print(f"✅ {intent} RAG智能体初始化成功")
                except Exception as e:
                    print(f"❌ {intent} RAG智能体初始化失败: {e}")
                    return None
            else:
                return None
        return self.rag_agents.get(intent)

    def process_question_with_full_response(self, user_input: str, stream_mode: bool = True):
        try:

            enhancement_result = enhancer.enhance_query(user_input)
            if enhancement_result and enhancement_result.get("intent_distribution"):
                distribution = enhancement_result["intent_distribution"]
                debug_parts = [f"{intent}: {count}" for intent, count in distribution.items()]
                print(f"🔍 [调试信息] 意图分布: {', '.join(debug_parts)}")

            if not enhancement_result or not enhancement_result.get("analysis_results"):
                if stream_mode: return self._stream_error("抱歉，未能识别出您问题的意图。")
                return [{"success": False, "message": "未能识别出意图"}]

            if stream_mode:
                return self._stream_answers_for_intents(enhancement_result)
            else:
                return self._get_batch_answers_for_intents(enhancement_result)

        except Exception as e:
            if stream_mode: return self._stream_error(f"处理过程中发生严重错误: {str(e)}")
            return [{"success": False, "message": f"处理过程中发生严重错误: {str(e)}"}]

    def _get_batch_answers_for_intents(self, enhancement_result: dict) -> list:
        all_responses = []
        original_query = enhancement_result.get("original_query")
        for item in enhancement_result["analysis_results"]:
            if "error" in item: continue
            Rag_intent = item["intent"]
            avatar = self.intent_avatar_mapping.get(Rag_intent, self.intent_avatar_mapping["其他"])
            try:
                if Rag_intent == "校园知识问答助手":
                    string_generator = self.llm.retrieve_and_answer(original_query, top_k=8)
                    answer = "".join(string_generator)
                else:
                    rag_agent = self.get_rag_agent(Rag_intent)
                    answer = rag_agent.call_RAG(original_query) if rag_agent else "抱歉，暂不支持此意图。"
                all_responses.append({"success": True, "intent": Rag_intent, "avatar": avatar, "answer": answer})
            except Exception as e:
                all_responses.append({"success": False, "intent": Rag_intent, "avatar": avatar, "error": str(e)})
        return all_responses

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

                try:
                    if Rag_intent == "校园知识问答助手":
                        paragraph_generator = self.llm.retrieve_and_answer(original_query, top_k=8)
                    else:
                        rag_agent = self.get_rag_agent(Rag_intent)
                        paragraph_generator = rag_agent.call_RAG_stream(original_query) if rag_agent else iter(["抱歉，暂不支持此意图。"])

                    for paragraph in paragraph_generator:
                        yield {"type": "content", "intent": Rag_intent, "avatar": avatar, "delta": paragraph}
                except Exception as e:
                    yield {"type": "error", "intent": Rag_intent, "avatar": avatar, "message": f"处理时发生错误: {str(e)}"}

                yield {"type": "break", "message": f"意图 {Rag_intent} 回答结束"}
            yield {"type": "finished", "finished": True}
        except Exception as e:
            yield from self._stream_error(f"流式处理时发生严重错误: {str(e)}")

    def _stream_error(self, message: str):
        yield {"type": "error", "message": message}
        yield {"type": "finished", "finished": True}

    def predict_intent_only(self, user_input):
        try:
            enhancement_result = enhancer.enhance_query(user_input)
            if not enhancement_result or not enhancement_result.get("analysis_results"):
                return {"success": False, "results": [], "message": "未能识别出任何意图"}

            identified_intents = []
            for item in enhancement_result["analysis_results"]:
                if "error" in item:
                    continue
                Rag_intent = item["intent"]
                avatar = self.intent_avatar_mapping.get(Rag_intent, self.intent_avatar_mapping["其他"])
                identified_intents.append({"intent": Rag_intent, "avatar": avatar})

            if not identified_intents:
                return {"success": False, "results": [], "message": "未能识别出任何有效意图"}

            return {"success": True, "results": identified_intents, "message": f"成功识别出 {len(identified_intents)} 个意图"}
        except Exception as e:
            return {"success": False, "results": [], "error": str(e), "message": "意图识别过程中发生未知错误"}


# 如果你想单独运行此文件进行命令行测试，可以保留这段代码
if __name__ == "__main__":
    try:
        agent = InteractiveAgent()
        # 简单测试
        print("命令行测试启动。输入 'exit' 退出。")
        while True:
            q = input("你: ")
            if q.lower() == 'exit':
                break
            for r in agent.process_question_with_full_response(q):
                print(r)
    except Exception as e:
        print(f"程序运行失败: {e}")
