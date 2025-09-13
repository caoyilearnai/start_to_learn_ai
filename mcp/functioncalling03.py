from openai import OpenAI
from datetime import datetime
import json
import os
import random

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
)

# 定义工具列表，模型在选择使用哪个工具时会参考工具的name和description
tools = [
    # 工具1 获取当前时刻的时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            # 因为获取当前时间无需输入参数，因此parameters为空字典
            "parameters": {},
        },
    },
    # 工具2 获取指定城市的天气
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    # 查询天气时需要提供位置，因此参数设置为location
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。",
                    }
                },
                "required": ["location"],
            },
        },
    },
]
import requests


def get_current_weather(arguments):
    # 从 JSON 中提取位置信息
    city_name = arguments["location"]
    """
    使用高德地图 API 查询天气
    """
    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        'key': os.getenv("GAODE_API_KEY"),
        'city': city_name,
        'extensions': 'base',  # base: 实况天气 | all: 预报
        'output': 'json'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data['status'] == '1' and len(data['lives']) > 0:
            w = data['lives'][0]
            result = {
                "城市": w['city'],
                "天气": w['weather'],
                "温度(°C)": w['temperature'],
                "风向": w['winddirection'],
                "风力": w['windpower'],
                "湿度(%)": w['humidity'],
                "发布时间": w['reporttime']
            }
            return  json.dumps(result)
        else:
            print("❌ 查询失败:", data.get('info', 'Unknown'))
            return None
    except Exception as e:
        print("❌ 请求失败:", e)
        return None


# 查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“
def get_current_time():
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 格式化当前日期和时间
    formatted_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    # 返回格式化后的当前时间
    return f"当前时间：{formatted_time}。"


# 封装模型响应函数
def get_response(messages):
    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        tools=tools,
    )
    return completion


def call_with_messages():
    print("\n")
    messages = [
        {
            "content": input(
                "请输入："
            ),  # 提问示例："现在几点了？" "一个小时后几点" "北京天气如何？"
            "role": "user",
        }
    ]
    print("-" * 60)
    # 模型的第一轮调用
    i = 1
    first_response = get_response(messages)
    assistant_output = first_response.choices[0].message
    print(f"\n第{i}轮大模型输出信息：{first_response}\n")
    if assistant_output.content is None:
        assistant_output.content = ""
    messages.append(assistant_output)
    # 如果不需要调用工具，则直接返回最终答案
    if (
        assistant_output.tool_calls == None
    ):  # 如果模型判断无需调用工具，则将assistant的回复直接打印出来，无需进行模型的第二轮调用
        print(f"无需调用工具，我可以直接回复：{assistant_output.content}")
        return
    # 如果需要调用工具，则进行模型的多轮调用，直到模型判断无需调用工具
    while assistant_output.tool_calls != None:
        # 如果判断需要调用查询天气工具，则运行查询天气工具
        tool_info = {
            "content": "",
            "role": "tool",
            "tool_call_id": assistant_output.tool_calls[0].id,
        }
        if assistant_output.tool_calls[0].function.name == "get_current_weather":
            # 提取位置参数信息
            arguments = json.loads(assistant_output.tool_calls[0].function.arguments)
            tool_info["content"] = get_current_weather(arguments)
        # 如果判断需要调用查询时间工具，则运行查询时间工具
        elif assistant_output.tool_calls[0].function.name == "get_current_time":
            tool_info["content"] = get_current_time()
        tool_output = tool_info["content"]
        print(f"工具输出信息：{tool_output}\n")
        print("-" * 60)
        messages.append(tool_info)
        assistant_output = get_response(messages).choices[0].message
        if assistant_output.content is None:
            assistant_output.content = ""
        messages.append(assistant_output)
        i += 1
        print(f"第{i}轮大模型输出信息：{assistant_output}\n")
    print(f"最终答案：{assistant_output.content}")


if __name__ == "__main__":
    call_with_messages()