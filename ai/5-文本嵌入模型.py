from langchain_community.embeddings import DashScopeEmbeddings

# 默认是text
model = DashScopeEmbeddings(dashscope_api_key="sk-9eea758856ee48f6a55418c9be3c29f5")

print(model.embed_query("我喜欢你"))

print("*"*50)

print(model.embed_documents(['我喜欢你','我也喜欢你！','我们在一起把']))