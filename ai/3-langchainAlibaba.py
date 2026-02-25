from langchain_community.llms.tongyi import Tongyi
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
import dashscope

DASHSCOPE_API_KEY = "sk-9eea758856ee48f6a55418c9be3c29f5"
dashscope.api_key = DASHSCOPE_API_KEY

model = Tongyi(model="qwen-plus", dashscope_api_key=DASHSCOPE_API_KEY)

response = model.stream(input="给我写一份唐诗")

for i in response:
    print(i,end="",flush=True)

