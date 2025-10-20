from dashscope import Application
from http import HTTPStatus
import os


class LLM_model:
    def __init__(self, app_id=None):
        self.session_id = "default_session"
        self.app_id = app_id or os.getenv("LLM_appid") or 'a9865e510ef540f08774d0ae31d22ad2'
        self.api_key = os.getenv("BAILIAN_API_KEY") or 'sk-93817db303964020bbc79b017be4768b'

        if not self.api_key:
            raise ValueError("请设置 BAILIAN_API_KEY")
        if not self.app_id:
            raise ValueError("请提供 app_id 或设置 LLM_appid")

    def start_LLM(self):
        return f"LLM model started successfully with APP_ID: {self.app_id}"

    def get_system_prompt(self):
        return """你是一位知识助手，请根据用户的问题和下列片段生成准确简洁的回答。请不要在回答中使用任何表情符号。 """

    def get_stream_system_prompt(self):
        """
        获取流式输出的系统提示词。
        --- 修改部分：加入分段和分隔符指令 ---
        """
        return """你是一位知识助手，你需要模仿人类的对话风格简洁输出，将你的回答拆分成3到5个自然段落，首先要确保你的回答拼起来是连贯的，符合人类讲出来的，其次是段与段之间要在语义和逻辑上相互承接。每个段落结束后，必须使用特殊标记 `[NEW_PARAGRAPH]` 作为换段标志。请不要在回答中使用任何表情符号。字数控制在100字以内"""

    def call_llm(self, query, list) -> str:
        separator = "\n\n"
        system_prompt = self.get_system_prompt()
        prompt = f"""{system_prompt}

用户问题: {query}

相关片段:
{separator.join(list)}

请基于上述内容作答，不要编造信息，不要使用表情符号。"""

        resp = Application.call(api_key=self.api_key, app_id=self.app_id, prompt=prompt, session_id=self.session_id)
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"API调用失败: {resp}")
        return resp.output.text

    def call_llm_stream(self, query, list):
        """
        流式生成回答，按段落返回。

        Args:
            query (str): 用户问题
            list (list): 相关文档片段列表

        Yields:
            str: 生成的文本段落
        """
        separator = "\n\n"
        system_prompt = self.get_stream_system_prompt()
        prompt = f"""{system_prompt}

请根据用户的问题和下面的背景知识进行简洁回答。

用户问题: {query}

背景知识:
{separator.join(list)}

若用户问题与背景知识无关，则用通用知识解决问题。请记住，你的回答必须被 `[NEW_PARAGRAPH]` 分隔成3到5段。

请开始你的回答：
"""
        # --- 修改部分：调整流式逻辑为分段输出，也就是达到了拟人化的效果 ---
        # 1. 使用非流式调用获取完整回答
        full_response_text = ""
        try:
            response = Application.call(
                api_key=self.api_key,
                app_id=self.app_id,
                prompt=prompt,
                session_id=self.session_id,
                stream=False  # 改为False，获取完整响应
            )
            if response.status_code == HTTPStatus.OK:
                request_id = response.request_id
                print(f"成功获取到回答，Request ID: {request_id}")  # 你可以在这里打印或记录它
                full_response_text = response.output.text
            else:
                error_message = f'API Error: {response.code} {response.message}'
                print(error_message)
                yield error_message  # 将错误信息作为一段返回
                return
        except Exception as e:
            error_message = f"调用LLM时发生异常: {e}"
            print(error_message)
            yield error_message
            return

        # 2. 根据分隔符切分段落并依次返回
        paragraphs = full_response_text.split('[NEW_PARAGRAPH]')
        for para in paragraphs:
            cleaned_para = para.strip()
            if cleaned_para:  # 确保不返回空段落
                yield cleaned_para


# --- 修改部分：在这里我更新了所有子类的流式Prompt ---

class LLM_psychology(LLM_model):
    def get_stream_system_prompt(self):
        return """你是一个专业的心理健康助手。你需要模仿人类温暖、共情的对话风格进行简洁回答，例如“我得知了你现在的处境，我知道你现在很难过，我觉得...”这样的句式。并增加人类那般的语气词如“呢、噢、哦”等，并将你的回答拆分成3到5个自然段落，首先要确保你的回答拼起来是连贯的，符合人类讲出来的，其次是段与段之间要在语义和逻辑上相互承接。每个段落结束后，必须使用特殊标记 `[NEW_PARAGRAPH]` 作为换段标志。
请遵循以下原则：
1. 保持温暖、理解和同理心的语调。
2. 提供科学、专业的心理学建议。
3. 若用户问题与心理学背景知识无关，请用心理健康的角度和通用知识来解决问题。
4. 请不要使用表情符号。
5.字数控制在100字以内
"""


class LLM_fitness(LLM_model):
    def get_stream_system_prompt(self):
        return """你是一个专业的健身和营养助手。你需要模仿专业教练的对话风格，例如“你应该参照这个科学的计划进行训练、你可以参考这样的营养表进行增肌、你的训练计划应该...”这样的话语，将你的回答拆分成3到5个自然段落，首先要确保你的回答拼起来是连贯的，符合人类讲出来的，其次是段与段之间要在语义和逻辑上相互承接。每个段落结束后，必须使用特殊标记 `[NEW_PARAGRAPH]` 作为换段标志。
请遵循以下原则：
1. 提供科学、安全的健身和营养建议。
2. 强调循序渐进和个体化的重要性。
3. 若用户问题与健身营养背景知识无关，请用健康生活的角度和通用知识来解决问题。
4. 请不要使用表情符号。
5.字数控制在100字以内
"""


class LLM_compus(LLM_model):
    def get_stream_system_prompt(self):
        return """你是一个校园知识问答助手。你需要模仿学长学姐亲切的对话风格，例如"我们学校的体育馆是....、我们学校的饭堂是...我们学校绩点的计算是..."这样的话语将你的回答拆分成3到5个自然段落，首先要确保你的回答拼起来是连贯的，符合人类讲出来的，其次是段与段之间要在语义和逻辑上相互承接。每个段落结束后，必须使用特殊标记 `[NEW_PARAGRAPH]` 作为换段标志。
请遵循以下原则：
1. 提供准确、及时的校园信息。
2. 保持友好、耐心的服务态度。
3. 若用户问题与校园背景知识无关，请用学生服务的角度和通用知识来解决问题。
4. 请不要使用表情符号。
5.字数控制在100字以内"""


class LLM_paper(LLM_model):
    def get_stream_system_prompt(self):
        return """你是一个专业的学术论文写作助手。你需要模仿严谨的导师或学者的对话风格，例如"你的论文格式应该...、你的参考文献可以....、你可以通过..方式检索...领域的文献"将你的回答拆分成3到5个自然段落，首先要确保你的回答拼起来是连贯的，符合人类讲出来的，其次是段与段之间要在语义和逻辑上相互承接。每个段落结束后，必须使用特殊标记 `[NEW_PARAGRAPH]` 作为换段标志。
请遵循以下原则：
1. 提供专业、严谨的学术建议。
2. 遵循学术规范和引用标准。
3. 若用户问题与学术背景知识无关，请用学术研究的角度和通用知识来解决问题。
4. 请不要使用表情符号。
5.字数控制在100字以内"""
