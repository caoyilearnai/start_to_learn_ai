import os
import json
from openai import OpenAI

# 定义工具
tools = [
    {
        "name": "get_current_time",
        "description": "当你想知道现在的时间时非常有用。",
        "parameters": {}
    },
    {
        "name": "get_current_weather",
        "description": "当你想查询指定城市的天气时非常有用。",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市或县区，比如北京市、杭州市、余杭区等。",
                }
            },
            "required": ["location"]
        }
    }
]

tools_string = json.dumps(tools,ensure_ascii=False)

system_prompt = f"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You may call one or more tools to assist with the user query. The tools you can use are as follows:
{tools_string}
Response in INTENT_MODE."""
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
messages = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': "杭州天气"}
    ]
response = client.chat.completions.create(
    model="tongyi-intent-detect-v3",
    messages=messages
)

print(response.choices[0].message.content)