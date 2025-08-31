import os

from openai import OpenAI


def get_weather(location, date):
    return f'在{date},{location} 的天气是晴天。'

def get_response(messages):
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    completion = client.chat.completions.create(model="qwen-plus", messages=messages)
    return completion

prompt = "你是谁"
messages = [
    {
        "role": "system",
        "content": "你是一个问答助手",
    }
]

messages.append(
    {
        "role": "user",
        "content":f'{prompt}'
    })
res = get_response(messages).choices[0].message.content
print(res)