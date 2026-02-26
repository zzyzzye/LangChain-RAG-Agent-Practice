from typing import Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.prompt_values import StringPromptValue
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

"""
PromptTemplate -> StringPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
FewShotPromptTemplate -> StringPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
ChatPromptTemplate -> BaseChatPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
"""

template: PromptTemplate = PromptTemplate.from_template("我的邻居是：{lastname}，最喜欢：{hobby}")


res: str = template.format(lastname="张大明", hobby="钓鱼")
print(f"format 结果: {res}, 类型: {type(res)}")


input_data: Dict[str, str] = {"lastname": "周杰伦", "hobby": "唱歌"}
res2: StringPromptValue = template.invoke(input_data)
print(f"invoke 结果: {res2}, 类型: {type(res2)}")


res2_str: str = res2.to_string()
print(f"invoke 转字符串结果: {res2_str}, 类型: {type(res2_str)}")