from typing import List, Dict, Any
import dashscope
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.llms.tongyi import Tongyi

# 1. 定义示例数据的类型（Dict[str, str] 表示键值都是字符串的字典）
ExampleType = Dict[str, str]

# 2. 定义单条示例的模板（指定返回类型为 PromptTemplate）
def create_example_template() -> PromptTemplate:
    """创建单条示例的提示词模板"""
    return PromptTemplate.from_template("单词:{word}，反义词:{antonym}")

# 3. 示例数据（明确标注为 List[ExampleType] 类型）
example_data: List[ExampleType] = [
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"}
]

# 4. 构建 FewShot 提示词模板（标注变量类型和返回类型）
def build_few_shot_prompt(
    example_template: PromptTemplate,
    examples: List[ExampleType]
) -> FewShotPromptTemplate:
    """构建FewShot提示词模板"""
    few_shot_prompt: FewShotPromptTemplate = FewShotPromptTemplate(
        example_prompt=example_template,
        examples=examples,
        prefix="给出给定词的反义词，有如下示例：",
        suffix="基于示例告诉我：{input_word}的反义词是？",
        input_variables=['input_word']  
    )
    return few_shot_prompt

# 5. 生成提示词逻辑
def get_prompt() -> str:
    """创建并返回格式化后的提示词字符串"""
    example_template: PromptTemplate = create_example_template()
    few_shot_prompt: FewShotPromptTemplate = build_few_shot_prompt(example_template, example_data)
    
    # 生成最终提示词
    prompt_input: Dict[str, str] = {"input_word": "左"}
    # invoke 返回的是 PromptValue 对象，使用 to_string() 转为字符串
    prompt_text: str = few_shot_prompt.invoke(input=prompt_input).to_string()
    return prompt_text

def call_model(prompt_text: str) -> str:
    """调用模型并返回结果字符串"""
    DASHSCOPE_API_KEY: str = "sk-9eea758856ee48f6a55418c9be3c29f5"
    dashscope.api_key = DASHSCOPE_API_KEY
    
    # 实例化模型
    model: Tongyi = Tongyi(model="qwen-plus", dashscope_api_key=DASHSCOPE_API_KEY)
    
    # 调用模型并获取返回文本
    response: str = model.invoke(prompt_text)
    return response

# 执行主逻辑
if __name__ == "__main__":
    # 1. 获取生成的提示词
    text: str = get_prompt()
    print(f"--- 生成的提示词 ---\n{text}\n")
    
    # 2. 调用模型
    print("--- AI 的回答 ---")
    res: str = call_model(text)
    print(res)
