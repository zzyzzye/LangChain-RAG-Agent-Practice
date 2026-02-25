from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

import dashscope

# 定义提示词模板
prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，帮忙起名字，请简略回答。"
)

# 变量注入，生成提示词文本
prompt_text = prompt_template.format(lastname="张", gender="女儿")

DASHSCOPE_API_KEY = "sk-9eea758856ee48f6a55418c9be3c29f5"
dashscope.api_key = DASHSCOPE_API_KEY
model = Tongyi(model="qwen-plus", dashscope_api_key=DASHSCOPE_API_KEY)

res = model.invoke(input=prompt_text)

print(res)