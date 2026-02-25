import dashscope
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage

DASHSCOPE_API_KEY = "sk-9eea758856ee48f6a55418c9be3c29f5"
dashscope.api_key = DASHSCOPE_API_KEY

response = ChatTongyi(model='qwen-plus',dashscope_api_key=DASHSCOPE_API_KEY)

messages = [
    SystemMessage(content="你是一个边塞诗人。"),
    HumanMessage(content="写一首唐诗"),
    AIMessage(content="锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    HumanMessage(content="按照你上一个回复的格式，再写一首唐诗。")
]

# messages = [
#     ("system", "你是一个边塞诗人。"),
#     ("human", "写一首唐诗"),
#     ("ai", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
#     ("human", "按照你上一个回复的格式，再写一首唐诗。")
# ]

# 调用stream流式执行
res = model.stream(input=messages)

for chunk in res:
    print(chunk.content, end="", flush=True)