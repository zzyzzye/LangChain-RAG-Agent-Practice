from typing import List, Tuple, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import BaseMessage, AIMessage
import dashscope

# 1. 定义对话模板
chat_prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗。"),
        MessagesPlaceholder("history"),
        ("human", "请再来一首唐诗"),
    ]
)

# 2. 准备历史数据 (类型为包含元组的列表)
history_data: List[Tuple[str, str]] = [
    ("human", "你来写一个唐诗"),
    ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "好诗再来一个"),
    ("ai", "锄禾日当午，汗滴禾下锄，谁知盘中餐，粒粒皆辛苦"),
]

# 3. 注入数据并生成提示词对象
prompt_value: ChatPromptValue = chat_prompt_template.invoke({"history": history_data})

# 4. 配置模型
DASHSCOPE_API_KEY: str = "sk-9eea758856ee48f6a55418c9be3c29f5"
dashscope.api_key = DASHSCOPE_API_KEY
model: ChatTongyi = ChatTongyi(model="qwen-plus", dashscope_api_key=DASHSCOPE_API_KEY)

# 5. 调用模型
response: AIMessage = model.invoke(prompt_value)

# 6. 输出结果
print(response.content)