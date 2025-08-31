import os
import requests  # 用于调用真实天气API
from openai import OpenAI

# 1. 初始化客户端
client = OpenAI(
    # 若没有配置环境变量，请将下行替换为：api_key="sk-你的百炼API Key",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 定义可用的工具（函数）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "查询指定城市的当前天气状况。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，例如 北京、上海、杭州"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# 3. 用户初始消息
messages = [{"role": "user", "content": "杭州天气怎么样？"}]

# 4. 第一步：调用大模型，让它决定是否需要调用工具
print("第一步：询问模型是否需要调用工具...")
completion = client.chat.completions.create(
    model="qwen-plus",  # 可替换为 qwen-max, qwen-turbo 等
    messages=messages,
    tools=tools,
    tool_choice="auto"  # 让模型自动决定是否调用工具
)

# 获取模型响应
response = completion.choices[0].message

# 5. 检查模型是否返回了工具调用请求
if response.tool_calls:
    print(f"模型建议调用函数: {response.tool_calls[0].function.name}")
    print(f"参数: {response.tool_calls[0].function.arguments}")

    # 5.1 提取要调用的函数名和参数
    tool_call = response.tool_calls[0]
    function_name = tool_call.function.name
    arguments = tool_call.function.arguments  # 这是JSON字符串

    # 5.2 执行真实的天气查询（这里使用 Open-Meteo 免费API）
    if function_name == "get_current_weather":
        import json
        args_dict = json.loads(arguments)  # 将JSON字符串转为字典
        city = args_dict.get("location", "unknown")

        print(f"\n第二步：正在查询 {city} 的真实天气...")

        # 使用 Open-Meteo API 获取真实天气
        # 注意：Open-Meteo 使用经纬度，这里我们先用城市名做简单处理（实际项目应结合地理编码）
        # 我们使用其“按城市名搜索”功能（非官方，但可用）或直接用已知坐标（杭州）
        # 更好的方式是使用 Geocoding API，但为简化，我们直接用杭州坐标
        # 杭州经纬度：30.2741, 120.1551
        weather_url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 30.2741,
            "longitude": 120.1551,
            "current": ["temperature_2m", "relative_humidity_2m", "weather_code", "wind_speed_10m"],
            "timezone": "Asia/Shanghai"
        }

        try:
            weather_response = requests.get(weather_url, params=params, timeout=10)
            weather_response.raise_for_status()
            weather_data = weather_response.json()

            # 提取关键信息
            current = weather_data.get("current", {})
            temp = current.get("temperature_2m", "未知")
            humidity = current.get("relative_humidity_2m", "未知")
            wind_speed = current.get("wind_speed_10m", "未知")
            # 天气代码映射（简化版）
            weather_code_desc = {
                0: "晴天", 1: "局部多云", 2: "多云", 3: "阴天",
                45: "雾", 48: "雾", 51: "毛毛雨", 61: "小雨",
                63: "中雨", 65: "大雨", 80: "阵雨", 81: "中阵雨"
            }
            wmo_code = current.get("weather_code", -1)
            condition = weather_code_desc.get(wmo_code, f"代码 {wmo_code}")

            # 构造自然语言结果
            real_weather_result = (
                f"{city} 当前天气：{condition}，"
                f"气温 {temp}°C，"
                f"湿度 {humidity}%，"
                f"风速 {wind_speed} km/h。"
            )
            print(f"真实天气数据: {real_weather_result}")

            # 5.3 将真实结果以工具调用响应的形式添加到消息历史
            messages.append(response)  # 添加模型的工具调用请求
            messages.append({
                "role": "tool",
                "content": real_weather_result,  # 将真实结果作为工具执行结果
                "tool_call_id": tool_call.id  # 必须匹配
            })

            # 6. 第三步：再次调用模型，让它基于真实天气数据生成最终回复
            print("\n第三步：让模型生成最终回复...")
            final_completion = client.chat.completions.create(
                model="qwen-plus",
                messages=messages  # 包含完整上下文
            )
            final_response = final_completion.choices[0].message.content
            print(f"\n🤖 最终回复: {final_response}")

        except requests.RequestException as e:
            error_msg = f"获取天气数据失败: {str(e)}"
            print(error_msg)
            # 也可以将错误信息反馈给模型
            messages.append(response)
            messages.append({
                "role": "tool",
                "content": error_msg,
                "tool_call_id": tool_call.id
            })
            # 再次调用模型告知用户失败
            final_completion = client.chat.completions.create(
                model="qwen-plus",
                messages=messages
            )
            print(f"\n🤖 模型回复: {final_completion.choices[0].message.content}")

else:
    # 模型认为无需调用工具，直接回复
    print(f"模型直接回复: {response.content}")