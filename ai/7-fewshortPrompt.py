from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 定义单条示例的模板
example_template = PromptTemplate.from_template("单词:{word}，反义词:{antonym}")

# 示例数据
example_data = [
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"}
]

# 构建 FewShot 提示词模板
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_template,
    examples=example_data,
    prefix="给出给定词的反义词，有如下示例：",
    suffix="基于示例告诉我：{input_word}的反义词是？",
    input_variables=['input_word']
)

# 生成最终提示词
prompt_text = few_shot_prompt.invoke(input={"input_word": "左"}).to_string()
print(prompt_text)