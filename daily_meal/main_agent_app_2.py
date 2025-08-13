# main_agent_app.py
import os
import json
import re
from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# 确保tools.py在Python路径中，或者与此文件在同一目录下
import tools as db_tools

# 导入LangChain和LangGraph相关模块
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

# LangGraph
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Union
import operator

from env_utils import DEEPSEEK_API_KEY

# 设置API Key - 推荐使用环境变量
# os.environ["DEEPSEEK_API_KEY"] = "YOUR_DEEPSEEK_API_KEY"
# 或者 os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # 如果使用OpenAI模型

# 假设的用户ID，实际应用中应从用户认证中获取
CURRENT_USER_ID = 12


# ==============================================================================
# 1. 定义状态 (Graph State)
# ==============================================================================
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    current_intent: str
    meal_info: dict
    pending_meal_confirmation: dict
    error_message: str
    tool_output: str
    requires_confirmation: bool


# ==============================================================================
# 2. 定义工具 (Tools)
# ==============================================================================
@tool
def add_food(name: str, description: str = "") -> str:
    """
    添加一个新的食品到库存。
    :param name: 食品名称。
    :param description: 食品描述（可选）。
    :return: JSON 格式的操作结果。
    """
    result = db_tools.add_food_tool(user_id=CURRENT_USER_ID, name=name, description=description)
    return json.dumps(result, ensure_ascii=False)


@tool
def get_food_by_name(food_name: str) -> str:
    """
    根据食品名称查询食品库存。
    :param food_name: 要查询的食品名称。
    :return: JSON 格式的查询结果列表。
    """
    result = db_tools.get_food_by_name_tool(user_id=CURRENT_USER_ID, name=food_name)
    return json.dumps(result, ensure_ascii=False)


@tool
def add_meal_record(meal_type: str, food_id: int, quantity: float) -> str:
    """
    添加用户的一顿餐食记录。
    :param meal_type: 用餐类型 (如 '早餐', '午餐', '晚餐', '加餐')。
    :param food_id: 关联的食品ID。
    :param quantity: 食用数量 (如克、份、毫升)。
    :return: JSON 格式的操作结果。
    """
    result = db_tools.add_meal_record_tool(user_id=CURRENT_USER_ID, meal_type=meal_type, food_id=food_id,
                                           quantity=quantity)
    return json.dumps(result, ensure_ascii=False)


@tool
def get_meal_records(meal_type: str = None, start_date: str = None, end_date: str = None) -> str:
    """
    获取用户的餐食记录。
    :param meal_type: 可选的餐食类型过滤 (如 '早餐', '午餐')。
    :param start_date: 开始日期 (YYYY-MM-DD)。
    :param end_date: 结束日期 (YYYY-MM-DD)。
    :return: JSON 格式的餐食记录列表。
    """
    result = db_tools.get_meal_records_tool(user_id=CURRENT_USER_ID, meal_type=meal_type, start_date=start_date,
                                            end_date=end_date)
    return json.dumps(result, ensure_ascii=False)


@tool
def update_meal_record(record_id: int, meal_type: str = None, food_id: int = None, quantity: float = None) -> str:
    """
    更新用户的餐食记录。
    :param record_id: 要更新的餐食记录ID。
    :param meal_type: 新的用餐类型（可选）。
    :param food_id: 新的食品ID（可选）。
    :param quantity: 新的食用数量（可选）。
    :return: JSON 格式的操作结果。
    """
    result = db_tools.update_meal_record_tool(record_id=record_id, user_id=CURRENT_USER_ID, meal_type=meal_type,
                                              food_id=food_id, quantity=quantity)
    return json.dumps(result, ensure_ascii=False)


@tool
def get_meal_record_by_id(record_id: int) -> str:
    """
    根据记录ID获取单个餐食记录的详细信息。
    :param record_id: 餐食记录ID。
    :return: JSON 格式的餐食记录详情。
    """
    result = db_tools.get_meal_record_by_id_tool(record_id=record_id, user_id=CURRENT_USER_ID)
    return json.dumps(result, ensure_ascii=False)


all_tools = [add_food, get_food_by_name, add_meal_record, get_meal_records, update_meal_record, get_meal_record_by_id]

# ==============================================================================
# 3. 初始化 LLM
# ==============================================================================
llm = ChatOpenAI(
    temperature=0,
    model='deepseek-chat',
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com")


# ==============================================================================
# 4. 定义 Agent (Nodes in LangGraph)
# ==============================================================================

def create_agent(llm, tools, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return executor


# --- 4.1 主 Agent (RouterAgent) ---
system_router_prompt = """你是一个智能路由助手，根据用户的请求内容，判断其核心意图。
你的任务是严格按照以下规则，将用户的请求精确分类：
- 'food_management': 用户的请求涉及添加新食品到库存、查询现有食品种类或数量。
- 'meal_management': 用户的请求涉及记录一餐、查询餐食记录、修改或删除餐食记录。
- 'greeting': 用户只是进行简单的问候，不包含具体操作意图。
- 'confirmation': 用户明确表示同意、确认或“是”，通常是对上一轮提问的肯定回应。
- 'rejection': 用户明确表示不同意、取消或“否”，通常是对上一轮提问的否定回应。
- 'unknown': 你的指令不明确，或者用户输入无法归类到以上任何一种意图。

**重要！你的回答必须是一个且仅是一个JSON字符串，不包含任何额外文本或解释。**
JSON结构必须严格如下所示，它是一个包含单个键 "intent" 的对象，其值是上述分类名称之一。例如：
{{ "intent": "food_management" }}
请不要在你的输出中包含任何 markdown 代码块标识符，例如 ```json 或 ```。只需输出纯粹的JSON字符串。

以下是更多示例，帮助你理解如何分类和给出响应：
用户输入 "我想吃苹果" -> {{ "intent": "meal_management" }}
用户输入 "添加一个香蕉到库存" -> {{ "intent": "food_management" }}
用户输入 "你好" -> {{ "intent": "greeting" }}
用户输入 "是的" -> {{ "intent": "confirmation" }}
用户输入 "取消" -> {{ "intent": "rejection" }}
用户输入 "我不知道" -> {{ "intent": "unknown" }}
"""

router_llm = llm
router_prompt = ChatPromptTemplate.from_messages([
    ("system", system_router_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


# 路由节点
def route_user_input(state: AgentState) -> AgentState:
    print(f"--- 路由中: {state['input']}")

    intent = "unknown"
    error_message_for_state = ""

    try:
        response_content = router_llm.invoke(
            router_prompt.format_prompt(
                input=state["input"],
                chat_history=state["chat_history"]
            ).to_messages()
        ).content
        print(f"LLM原始输出: {response_content}")

        match = re.search(r"```json\s*(.*?)\s*```", response_content, re.DOTALL)

        json_to_parse = ""
        if match:
            json_to_parse = match.group(1).strip()
            print(f"通过正则清理后的字符串: '{json_to_parse}'")
        else:
            json_to_parse = response_content.strip()
            print(f"未检测到Markdown JSON，使用原始内容: '{json_to_parse}'")

        try:
            intent_json = json.loads(json_to_parse)
            if "intent" in intent_json:
                intent = intent_json["intent"]
                print(f"解析成功！意图: {intent}")
            else:
                print(f"JSON中未找到'intent'字段: {intent_json}")
                intent = "unknown"
                error_message_for_state = f"路由LLM输出JSON缺少'intent'字段。原始输出: {response_content}"
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}. 尝试解析的字符串: '{json_to_parse}'")
            intent = "unknown"
            error_message_for_state = f"路由LLM输出格式错误。原始输出: {response_content}"

    except Exception as e:
        print(f"路由LLM调用失败: {e}")
        intent = "unknown"
        error_message_for_state = f"路由LLM调用失败: {e}"

    return {"current_intent": intent, "error_message": error_message_for_state}


# --- 4.2 食品管理 Agent (FoodManagementAgent) ---
system_food_prompt = """你是一个专业的食品库存管理助手。
你可以执行以下操作：
1. 添加新食品：当用户要求添加食品时，调用 `add_food` 工具。
2. 查询食品：当用户询问某种食品是否存在时，调用 `get_food_by_name` 工具。
3. 如果用户要求添加食品，但没有提供食品名称，询问用户。
4. 始终以JSON格式输出你的最终响应，包含 'response_type' 和 'message' 字段。
   - 'response_type': "info", "success", "error", "prompt_food_name"
   - 'message': 具体的文本内容。
   - 如果是查询结果，请在message中包含食品信息。
   - 如果需要用户输入食品名称，response_type为"prompt_food_name"。
"""
food_management_agent_executor = create_agent(llm, [add_food, get_food_by_name], system_food_prompt)


def run_food_management_agent(state: AgentState) -> AgentState:
    print(f"--- 进入食品管理Agent: {state['input']}")
    result = food_management_agent_executor.invoke({
        "input": state["input"],
        "chat_history": state["chat_history"]
    })

    # 修正：在处理 AgentExecutor 的 output 时，也进行 Markdown 代码块提取
    if result.get("output") is not None:
        raw_llm_output = result["output"]
        match = re.search(r"```json\s*(.*?)\s*```", raw_llm_output, re.DOTALL)
        json_to_parse = ""
        if match:
            json_to_parse = match.group(1).strip()
            print(f"食品管理Agent：通过正则清理后的字符串: '{json_to_parse}'")
        else:
            json_to_parse = raw_llm_output.strip()
            print(f"食品管理Agent：未检测到Markdown JSON，使用原始内容: '{json_to_parse}'")

        try:
            parsed_output = json.loads(json_to_parse)
            state["tool_output"] = json.dumps(parsed_output, ensure_ascii=False)  # 存储解析后的JSON
            state["error_message"] = ""

            display_message = parsed_output.get("message", "Agent响应解析失败。")
            state["chat_history"].append(AIMessage(content=display_message))

            if parsed_output.get("response_type") == "prompt_food_name":
                state["requires_confirmation"] = False
            return state
        except json.JSONDecodeError:
            error_msg = f"食品管理助手输出非JSON。原始输出: {raw_llm_output}"
            state["tool_output"] = json.dumps({"response_type": "error", "message": error_msg}, ensure_ascii=False)
            state["error_message"] = error_msg
            state["requires_confirmation"] = False
            state["chat_history"].append(AIMessage(content=error_msg))
            return state

    if result.get("tool_calls"):
        state["chat_history"].append(AIMessage(content="", tool_calls=result["tool_calls"]))

        tool_output_content = result.get("tool_output", "工具没有返回结果")
        state["tool_output"] = tool_output_content  # 保存原始工具输出
        state["error_message"] = ""

        try:
            tool_output_parsed = json.loads(tool_output_content)
            display_tool_message = tool_output_parsed.get("message", tool_output_content)
        except json.JSONDecodeError:
            display_tool_message = tool_output_content

        state["chat_history"].append(ToolMessage(
            content=display_tool_message,
            tool_call_id=result["tool_calls"][0]["id"] if result["tool_calls"] else "unknown_tool_call"
        ))
        state["requires_confirmation"] = False
    else:
        error_msg = "食品管理Agent未能生成有效响应（无output也无tool_calls）。"
        state["tool_output"] = json.dumps({"response_type": "error", "message": error_msg}, ensure_ascii=False)
        state["error_message"] = error_msg
        state["chat_history"].append(AIMessage(content=error_msg))
        state["requires_confirmation"] = False

    return state


# --- 4.3 餐食管理 Agent (MealManagementAgent) ---
system_meal_prompt = """你是一个智能餐食记录助手。
你的目标是帮助用户记录、查询和管理他们的一日三餐。
你可以执行以下操作：
1. **添加餐食记录**: 当用户想要记录一餐时，你需要收集以下信息：
   - 用餐类型 (meal_type): 早餐, 午餐, 晚餐, 加餐。
   - 食品名称 (food_name): 食用的具体食品。
   - 数量 (quantity): 食用的量，可以是整数或浮点数。
   - 如果信息不完整，你需要明确地询问用户缺失的信息。
   - 在添加前，你需要调用 `get_food_by_name` 工具检查该食品是否存在于库存中。
     - 如果存在，使用其 `food_id`。
     - 如果不存在，提示用户将该食品添加到库存。
   - 信息收集完成后，向用户展示完整信息，并要求用户确认。
   - 确认后，调用 `add_meal_record` 工具。
2. **查询餐食记录**: 当用户想要查询餐食记录时，调用 `get_meal_records` 工具。
   - 可以根据用餐类型或日期范围查询。
3. **更新餐食记录**: 当用户想在某一条记录上添加或替换食品种类和数量时，你需要：
   - 首先通过关键词或ID识别要修改的餐食记录。
   - 询问用户要添加/替换的食品名称和数量。
   - 引导用户提供`record_id`、`food_id`和`quantity`。
   - 调用 `update_meal_record` 工具。

**重要规则:**
- 每次与用户的交互都必须以JSON格式输出。
- JSON包含 `response_type` 和 `message` 字段。
- `response_type` 可以是:
    - `"prompt_meal_info"`: 需要用户提供餐食信息 (例如用餐类型、食品名称、数量)。
    - `"prompt_food_add"`: 提示用户将某个食品添加到食品库。
    - `"confirm_meal"`: 向用户展示收集到的完整餐食信息，请求确认。
    - `"meal_added_success"`: 餐食记录成功添加。
    - `"meal_updated_success"`: 餐食记录成功更新。
    - `"meal_query_result"`: 餐食查询结果。
    - `"info"`: 正常信息提示。
    - `"error"`: 错误信息。
    - `"prompt_record_id"`: 需要用户提供餐食记录ID。

**示例 JSON 响应:**
- 询问缺失信息: {{ "response_type": "prompt_meal_info", "message": "请告诉我您用餐类型、食品名称和数量。", "missing_info": ["meal_type", "food_name", "quantity"] }}
- 请求确认: {{ "response_type": "confirm_meal", "message": "您确定要记录：早餐，苹果，150克吗？", "meal_details": {{ "meal_type": "早餐", "food_name": "苹果", "quantity": 150 }} }}
- 成功添加: {{ "response_type": "meal_added_success", "message": "您的早餐记录（苹果，150克）已成功添加！" }}
- 查询结果: {{ "response_type": "meal_query_result", "message": "这是您的午餐记录：...", "records": [] }}
- 提示添加食品: {{ "response_type": "prompt_food_add", "message": "您的食品库中没有'牛奶'，请先将其添加到食品库：'添加食品 牛奶'" }}

记住，你的目标是逐步引导用户完成操作，必要时需多次交互以获取完整信息。
如果用户要求更新记录，但没有提供记录ID，请首先询问记录ID。
"""

meal_management_agent_executor = create_agent(llm,
                                              [add_meal_record, get_meal_records, get_food_by_name, update_meal_record,
                                               get_meal_record_by_id], system_meal_prompt)


# 餐食管理节点的函数
def run_meal_management_agent(state: AgentState) -> AgentState:
    print(f"--- 进入餐食管理Agent: {state['input']}")

    # --- 1. 处理确认/拒绝流程 (当 requires_confirmation 为 True 时，优先处理用户的确认/拒绝意图) ---
    if state.get("requires_confirmation") and state.get("pending_meal_confirmation"):
        if state["current_intent"] == "confirmation":
            meal_details = state["pending_meal_confirmation"]
            result = db_tools.add_meal_record_tool(
                user_id=CURRENT_USER_ID,
                meal_type=meal_details["meal_type"],
                food_id=meal_details["food_id"],
                quantity=meal_details["quantity"]
            )
            response_json_content = json.dumps(
                {"response_type": "meal_added_success", "message": result.get("message", "餐食已添加。")},
                ensure_ascii=False)

            state["pending_meal_confirmation"] = {}
            state["requires_confirmation"] = False
            state["meal_info"] = {}
            state["error_message"] = ""
            state["tool_output"] = response_json_content

            state["chat_history"].append(
                AIMessage(content=json.loads(response_json_content).get("message", "餐食操作完成。")))
            return state

        elif state["current_intent"] == "rejection":
            response_json_content = json.dumps(
                {"response_type": "info", "message": "好的，已取消餐食记录操作。请重新告诉我您的需求。"},
                ensure_ascii=False)

            state["pending_meal_confirmation"] = {}
            state["requires_confirmation"] = False
            state["meal_info"] = {}
            state["error_message"] = ""
            state["tool_output"] = response_json_content

            state["chat_history"].append(
                AIMessage(content=json.loads(response_json_content).get("message", "餐食操作已取消。")))
            return state

    # --- 2. 正常运行 Agent (如果不是确认/拒绝流程，或者确认/拒绝流程已处理完毕) ---
    result = meal_management_agent_executor.invoke({
        "input": state["input"],
        "chat_history": state["chat_history"],
        "meal_info": state["meal_info"]
    })

    # 修正：在处理 AgentExecutor 的 output 时，也进行 Markdown 代码块提取
    if result.get("output") is not None:
        raw_llm_output = result["output"]
        match = re.search(r"```json\s*(.*?)\s*```", raw_llm_output, re.DOTALL)
        json_to_parse = ""
        if match:
            json_to_parse = match.group(1).strip()
            print(f"餐食管理Agent：通过正则清理后的字符串: '{json_to_parse}'")
        else:
            json_to_parse = raw_llm_output.strip()
            print(f"餐食管理Agent：未检测到Markdown JSON，使用原始内容: '{json_to_parse}'")

        try:
            parsed_output = json.loads(json_to_parse)
            state["tool_output"] = json.dumps(parsed_output, ensure_ascii=False)  # 存储解析后的JSON
            state["error_message"] = ""

            display_message = parsed_output.get("message", "Agent响应解析失败。")
            state["chat_history"].append(AIMessage(content=display_message))

            if parsed_output.get("response_type") == "confirm_meal":
                state["pending_meal_confirmation"] = parsed_output.get("meal_details", {})
                state["requires_confirmation"] = True
            elif parsed_output.get("response_type") == "prompt_meal_info":
                if "collected_info" in parsed_output:
                    state["meal_info"].update(parsed_output["collected_info"])
                state["requires_confirmation"] = False
            else:
                state["requires_confirmation"] = False

            return state
        except json.JSONDecodeError:
            error_msg = f"餐食管理助手输出非JSON。原始输出: {raw_llm_output}"
            state["tool_output"] = json.dumps({"response_type": "error", "message": error_msg}, ensure_ascii=False)
            state["error_message"] = error_msg
            state["requires_confirmation"] = False
            state["chat_history"].append(AIMessage(content=error_msg))
            return state

    if result.get("tool_calls"):
        state["chat_history"].append(AIMessage(content="", tool_calls=result["tool_calls"]))
        tool_output_content = result.get("tool_output", "工具没有返回结果")
        state["tool_output"] = tool_output_content  # 保存原始工具输出
        state["error_message"] = ""

        try:
            tool_output_parsed = json.loads(tool_output_content)
            display_tool_message = tool_output_parsed.get("message", tool_output_content)
        except json.JSONDecodeError:
            display_tool_message = tool_output_content

        state["chat_history"].append(ToolMessage(
            content=display_tool_message,
            tool_call_id=result["tool_calls"][0]["id"] if result["tool_calls"] else "unknown_tool_call"
        ))
        state["requires_confirmation"] = False
    else:
        error_msg = "餐食管理Agent未能生成有效响应（无output也无tool_calls）。"
        state["tool_output"] = json.dumps({"response_type": "error", "message": error_msg}, ensure_ascii=False)
        state["error_message"] = error_msg
        state["chat_history"].append(AIMessage(content=error_msg))
        state["requires_confirmation"] = False

    return state


# --- 4.4 问候和未知意图处理 ---
def handle_greeting(state: AgentState) -> AgentState:
    print("--- 处理问候 ---")
    state["tool_output"] = json.dumps({"response_type": "info",
                                       "message": "您好！我是您的智能餐食助手，请问有什么可以帮助您的吗？想记录用餐？还是管理食品？"},
                                      ensure_ascii=False)
    state["error_message"] = ""
    state["requires_confirmation"] = False
    return state


def handle_unknown(state: AgentState) -> AgentState:
    print("--- 处理未知意图 ---")
    state["tool_output"] = json.dumps(
        {"response_type": "error", "message": "抱歉，我没有理解您的意思。请问您是想记录餐食，还是管理食品呢？"},
        ensure_ascii=False)
    state["error_message"] = "未知意图"
    state["requires_confirmation"] = False
    return state


# ==============================================================================
# 5. 构建 LangGraph
# ==============================================================================
workflow = StateGraph(AgentState)

# 定义节点
workflow.add_node("router", route_user_input)
workflow.add_node("food_management", run_food_management_agent)
workflow.add_node("meal_management", run_meal_management_agent)
workflow.add_node("greeting_handler", handle_greeting)
workflow.add_node("unknown_handler", handle_unknown)

# 定义入口
workflow.set_entry_point("router")

# 定义路由逻辑
workflow.add_conditional_edges(
    "router",
    lambda state: state["current_intent"],
    {
        "food_management": "food_management",
        "meal_management": "meal_management",
        "greeting": "greeting_handler",
        "confirmation": "meal_management",
        "rejection": "meal_management",
        "unknown": "unknown_handler",
    }
)

# 定义每个Agent执行完后的下一个步骤
workflow.add_edge("food_management", END)
workflow.add_edge("greeting_handler", END)
workflow.add_edge("unknown_handler", END)

workflow.add_conditional_edges(
    "meal_management",
    lambda state: "router" if state.get("requires_confirmation") else END,
    {
        "router": "router",  # 如果需要确认，再次通过 router 接收用户输入
        END: END  # 否则，结束本轮对话
    }
)

memory_checkpointer = MemorySaver()

# 指定 SQLite 文件路径（自动创建或连接）
# memory_checkpointer = SqliteSaver('sqlite:///checkpoints.db')
# 编译图
app = workflow.compile(checkpointer=memory_checkpointer)
thread_id = input('请输入一个sessionId (例如 user_id): ')
config = {"configurable": {"thread_id": thread_id}}


# ==============================================================================
# 6. 运行应用
# ==============================================================================
def chat_with_agent(user_input: str, chat_history: List[BaseMessage] = None, current_intent: str = None,
                    meal_info: dict = None, pending_meal_confirmation: dict = None,
                    requires_confirmation: bool = False):
    """
    与Agent进行交互的函数。
    :param user_input: 用户输入。
    :param chat_history: 聊天历史。
    :param current_intent: 当前的意图，用于路由。
    :param meal_info: 收集到的餐食信息。
    :param pending_meal_confirmation: 待确认的餐食信息。
    :param requires_confirmation: 是否需要用户确认。
    :return: 包含响应和更新状态的字典。
    """
    if chat_history is None:
        chat_history = []
    if meal_info is None:
        meal_info = {}
    if pending_meal_confirmation is None:
        pending_meal_confirmation = {}

    updated_chat_history = list(chat_history)  # 创建副本
    updated_chat_history.append(HumanMessage(content=user_input))

    initial_state = AgentState(
        input=user_input,
        chat_history=updated_chat_history,
        current_intent=current_intent,
        meal_info=meal_info,
        pending_meal_confirmation=pending_meal_confirmation,
        error_message="",
        tool_output="",
        requires_confirmation=requires_confirmation
    )

    final_state_data = None
    all_stream_outputs = []
    try:
        for s in app.stream(initial_state, config=config):
            all_stream_outputs.append(s)
            print(f"DEBUG: Stream yielded: {s}")

            if "__end__" in s:
                final_state_data = s["__end__"]
                break

    except Exception as e:
        print(f"LangGraph stream 运行异常: {e}")
        final_state_data = initial_state
        final_state_data["tool_output"] = json.dumps({"response_type": "error", "message": f"系统运行异常: {e}"},
                                                     ensure_ascii=False)
        final_state_data["error_message"] = str(e)

    if final_state_data is None and all_stream_outputs:
        print(
            "Warning: LangGraph stream did not yield a final state with '__end__'. Using the last available intermediate state.")
        last_item = all_stream_outputs[-1]
        if isinstance(last_item, dict) and len(last_item) == 1:
            final_state_data = list(last_item.values())[0]
        else:
            final_state_data = last_item

    if final_state_data is None:
        print("Error: LangGraph stream yielded no usable states. Falling back to initial state.")
        final_state_data = initial_state
        final_state_data["tool_output"] = json.dumps({"response_type": "error", "message": "系统未响应或会话异常结束。"},
                                                     ensure_ascii=False)
        final_state_data["error_message"] = "Graph did not complete normally or no states were yielded."

    response_content = final_state_data.get("tool_output",
                                            json.dumps({"response_type": "error", "message": "系统错误，未能获取响应。"},
                                                       ensure_ascii=False))

    new_chat_history = final_state_data.get("chat_history", [])

    return {
        "response": json.loads(response_content),
        "updated_state": {
            "chat_history": new_chat_history,
            "current_intent": final_state_data.get("current_intent"),
            "meal_info": final_state_data.get("meal_info"),
            "pending_meal_confirmation": final_state_data.get("pending_meal_confirmation"),
            "requires_confirmation": final_state_data.get("requires_confirmation")
        }
    }


# ==============================================================================
# 7. 演示交互流程
# ==============================================================================
if __name__ == "__main__":
    current_chat_history = []
    current_meal_info = {}
    current_pending_meal_confirmation = {}
    current_intent = None
    requires_confirmation = False

    print("--- 欢迎使用智能餐食管理助手！ ---")
    print("您可以尝试：")
    print("1. 问候：'你好'")
    print("2. 添加食品：'添加一个苹果'")
    print("3. 查询食品：'我的食品库里有什么苹果吗？'")
    print("4. 记录餐食：'我早餐吃了苹果'")
    print("5. 记录餐食（信息不全）：'记录一顿饭'")
    print("6. 确认/取消：'是的' / '取消'")
    print("7. 更新餐食：'我想把昨天午餐的苹果换成香蕉'")

    while True:
        user_input = input("\n您好: ")
        if user_input.strip().lower() in ["退出", "exit", "quit"]:
            break

        response_data = chat_with_agent(
            user_input=user_input,
            chat_history=current_chat_history,
            current_intent=current_intent,
            meal_info=current_meal_info,
            pending_meal_confirmation=current_pending_meal_confirmation,
            requires_confirmation=requires_confirmation
        )

        current_chat_history = response_data["updated_state"]["chat_history"]
        current_intent = response_data["updated_state"]["current_intent"]
        current_meal_info = response_data["updated_state"]["meal_info"]
        current_pending_meal_confirmation = response_data["updated_state"]["pending_meal_confirmation"]
        requires_confirmation = response_data["updated_state"]["requires_confirmation"]

        response_json = response_data["response"]
        print(f"助手 ({response_json.get('response_type', '未知')}): {response_json.get('message', '无消息')}")
        if response_json.get("records"):
            for record in response_json["records"]:
                print(
                    f"  - ID: {record.get('id')}, 类型: {record.get('meal_type')}, 食物: {record.get('food_name')}, 数量: {record.get('quantity')}, 时间: {record.get('create_time')}")
        if response_json.get("meal_details"):
            details = response_json["meal_details"]
            print(
                f"  - 用餐类型: {details.get('meal_type')}, 食品: {details.get('food_name')}, 数量: {details.get('quantity')}")
        if response_json.get("missing_info"):
            print(f"  - 缺失信息: {', '.join(response_json['missing_info'])}")