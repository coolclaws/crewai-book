# 第 16 章　Tools 框架：BaseTool 与调用机制

工具（Tool）是 Agent 与外部世界交互的桥梁。一个 Agent 无论推理能力多强，如果不能读取文件、调用 API、搜索网络，就无法完成真正有价值的任务。CrewAI 设计了一套完整的 Tools 框架，从抽象基类到函数式装饰器，从缓存机制到使用量控制，从工具选择到错误恢复，构成了一个层次分明、扩展性强的工具生态。

本章将从源码层面深入分析这套框架的每一个关键组件。

## 16.1 工具框架总览

CrewAI 的 tools 模块位于 `crewai/tools/` 目录下，文件结构如下：

```
tools/
    __init__.py              # 导出 BaseTool, EnvVar, tool
    base_tool.py             # BaseTool 抽象基类 + Tool 类 + @tool 装饰器
    structured_tool.py       # CrewStructuredTool —— 统一的工具执行层
    tool_calling.py          # ToolCalling / InstructorToolCalling 数据模型
    tool_usage.py            # ToolUsage —— 工具调用的核心编排器
    tool_types.py            # ToolResult 数据类
    cache_tools/
        cache_tools.py       # CacheTools —— 缓存查询工具
    agent_tools/
        base_agent_tools.py  # BaseAgentTool —— Agent 委托基类
        agent_tools.py       # AgentTools —— 委托工具管理器
        delegate_work_tool.py# DelegateWorkTool
        ask_question_tool.py # AskQuestionTool
        add_image_tool.py    # AddImageTool
        read_file_tool.py    # ReadFileTool
    memory_tools.py          # RecallMemoryTool / RememberTool
    mcp_tool_wrapper.py      # MCPToolWrapper —— 按需连接的 MCP 工具
    mcp_native_tool.py       # MCPNativeTool —— 复用 Session 的 MCP 工具
```

这些组件之间的继承与组合关系可以用如下层次图表示：

```
BaseModel (Pydantic)
  └── BaseTool (ABC)            # 抽象基类，定义 _run / _arun
        ├── Tool[P, R]          # 包装普通函数的通用工具
        ├── BaseAgentTool       # Agent 委托工具基类
        │     ├── DelegateWorkTool
        │     └── AskQuestionTool
        ├── AddImageTool        # 图片添加工具
        ├── ReadFileTool        # 文件读取工具
        ├── RecallMemoryTool    # 记忆检索工具
        ├── RememberTool        # 记忆保存工具
        ├── MCPToolWrapper      # MCP 按需连接工具
        └── MCPNativeTool       # MCP 持久连接工具

CrewStructuredTool               # 独立类，不继承 BaseTool
                                 # 统一的执行接口层
```

## 16.2 BaseTool：工具的抽象基类

`BaseTool` 是整个工具体系的根基，定义在 `base_tool.py` 中。它同时继承了 Pydantic 的 `BaseModel` 和 Python 的 `ABC`，这意味着每个工具实例既是一个可序列化的数据模型，又必须实现特定的抽象方法。

### 16.2.1 核心字段定义

```python
class BaseTool(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        description="The unique name of the tool that clearly communicates its purpose."
    )
    description: str = Field(
        description="Used to tell the model how/when/why to use the tool."
    )
    args_schema: type[PydanticBaseModel] = Field(
        default=_ArgsSchemaPlaceholder,
        validate_default=True,
        description="The schema for the arguments that the tool accepts.",
    )
    cache_function: Callable[..., bool] = Field(
        default=lambda _args=None, _result=None: True,
    )
    result_as_answer: bool = Field(default=False)
    max_usage_count: int | None = Field(default=None)
    current_usage_count: int = Field(default=0)
```

这里有几个关键设计决策值得注意：

1. **`args_schema` 的自动推断**：默认值是一个内部 placeholder 类 `_ArgsSchemaPlaceholder`。当用户没有显式提供 schema 时，`_default_args_schema` 验证器会自动从 `_run` 方法的签名中推断参数模型。

2. **`cache_function`**：一个可选的回调函数，决定某次工具调用的结果是否应被缓存。默认实现始终返回 `True`，表示所有结果都缓存。

3. **`result_as_answer`**：当设置为 `True` 时，工具的返回值将直接作为 Agent 的最终回答，跳过后续的推理步骤。

4. **`max_usage_count`**：限制工具的最大使用次数，防止 Agent 陷入循环调用。

### 16.2.2 自动生成 args_schema

这是 BaseTool 最巧妙的设计之一。通过 Pydantic 的 `field_validator`，框架在实例化时自动从 `_run` 方法的函数签名中提取参数信息：

```python
@field_validator("args_schema", mode="before")
@classmethod
def _default_args_schema(cls, v):
    if v != cls._ArgsSchemaPlaceholder:
        return v

    run_sig = signature(cls._run)
    fields: dict[str, Any] = {}

    for param_name, param in run_sig.parameters.items():
        if param_name in ("self", "return"):
            continue
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue

        annotation = param.annotation if param.annotation != param.empty else Any

        if param.default is param.empty:
            fields[param_name] = (annotation, ...)    # 必填参数
        else:
            fields[param_name] = (annotation, param.default)  # 可选参数

    return create_model(f"{cls.__name__}Schema", **fields)
```

这段代码的逻辑清晰明了：检查 `_run` 方法的每个参数，跳过 `self`、`*args`、`**kwargs`，将剩余参数转化为 Pydantic `create_model` 的字段定义。如果 `_run` 没有显式参数，还会尝试从 `_arun` 推断。这样用户只需定义 `_run` 方法，框架就能自动生成完整的参数验证 schema。

### 16.2.3 run 与 _run 的双层结构

BaseTool 采用了经典的模板方法模式：

```python
def run(self, *args, **kwargs) -> Any:
    if not args:
        kwargs = self._validate_kwargs(kwargs)
    result = self._run(*args, **kwargs)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    self.current_usage_count += 1
    return result

@abstractmethod
def _run(self, *args, **kwargs) -> Any:
    """子类必须实现"""
```

`run` 是公开接口，负责参数验证、协程处理、使用计数等通用逻辑；`_run` 是子类需要实现的具体执行逻辑。同样的模式也应用于异步接口 `arun` / `_arun`。

`_validate_kwargs` 方法利用 `args_schema` 对传入参数进行验证和类型转换：

```python
def _validate_kwargs(self, kwargs):
    if self.args_schema is not None and self.args_schema.model_fields:
        validated = self.args_schema.model_validate(kwargs)
        return validated.model_dump()
    return kwargs
```

### 16.2.4 描述文本的自动生成

每个工具在初始化后会自动生成一段结构化的描述文本，包含工具名称、参数 JSON Schema 和用户提供的描述：

```python
def _generate_description(self) -> None:
    schema = generate_model_description(self.args_schema)
    args_json = json.dumps(schema["json_schema"]["schema"], indent=2)
    self.description = (
        f"Tool Name: {sanitize_tool_name(self.name)}\n"
        f"Tool Arguments: {args_json}\n"
        f"Tool Description: {self.description}"
    )
```

这段生成的描述文本会被注入到 LLM 的 prompt 中，帮助模型理解如何正确调用工具。

## 16.3 Tool 类与 @tool 装饰器

`Tool` 类是 `BaseTool` 的通用具体实现，用于包装普通的 Python 函数：

```python
class Tool(BaseTool, Generic[P, R]):
    func: Callable[P, R | Awaitable[R]]

    def _run(self, *args, **kwargs) -> R:
        return self.func(*args, **kwargs)

    async def _arun(self, *args, **kwargs) -> R:
        result = self.func(*args, **kwargs)
        if _is_awaitable(result):
            return await result
        raise NotImplementedError(...)
```

`Tool` 类持有一个 `func` 字段，在 `_run` 中直接调用。如果 `func` 是异步函数，`_arun` 会正确地 `await` 它。

### @tool 装饰器

`@tool` 装饰器提供了三种使用方式：

```python
# 方式 1：无参数装饰器
@tool
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

# 方式 2：自定义名称
@tool("custom_greeter")
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

# 方式 3：带选项
@tool(result_as_answer=True, max_usage_count=3)
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"
```

装饰器内部通过 overload 签名和运行时参数检查来区分这三种用法。核心实现是 `_make_with_name` 闭包：

```python
def _make_with_name(tool_name: str):
    def _make_tool(f):
        if f.__doc__ is None:
            raise ValueError("Function must have a docstring")

        func_sig = signature(f)
        fields = {}
        for param_name, param in func_sig.parameters.items():
            # ... 提取参数类型和默认值
            pass

        args_schema = create_model(class_name, **fields)
        return Tool(
            name=tool_name,
            description=f.__doc__,
            func=f,
            args_schema=args_schema,
            result_as_answer=result_as_answer,
            max_usage_count=max_usage_count,
        )
    return _make_tool
```

注意装饰器强制要求函数必须有 docstring —— 这不是随意的限制，而是因为 docstring 将作为工具描述提供给 LLM。没有描述的工具，LLM 无法理解何时以及如何使用它。

## 16.4 CrewStructuredTool：统一执行层

`CrewStructuredTool` 是框架中实际负责工具执行的核心类，定义在 `structured_tool.py` 中。它不继承 `BaseTool`，而是作为一个独立的执行适配层存在。

### 16.4.1 从函数创建工具

```python
class CrewStructuredTool:
    def __init__(self, name, description, args_schema, func, ...):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.func = func
        self._original_tool: BaseTool | None = None
        self._validate_function_signature()
```

`from_function` 类方法提供了从普通函数创建工具的便捷方式：

```python
@classmethod
def from_function(cls, func, name=None, description=None, ...):
    name = name or func.__name__
    description = description or inspect.getdoc(func)
    if infer_schema:
        schema = cls._create_schema_from_function(name, func)
    return cls(name=name, description=description, args_schema=schema, func=func)
```

`_create_schema_from_function` 使用 `inspect.signature` 和 `get_type_hints` 从函数签名中动态创建 Pydantic 模型。

### 16.4.2 invoke 与参数解析

`invoke` 是 CrewStructuredTool 的核心执行方法：

```python
def invoke(self, input: str | dict, config=None, **kwargs):
    parsed_args = self._parse_args(input)

    if self.has_reached_max_usage_count():
        raise ToolUsageLimitExceededError(...)

    self._increment_usage_count()

    if inspect.iscoroutinefunction(self.func):
        return asyncio.run(self.func(**parsed_args, **kwargs))

    result = self.func(**parsed_args, **kwargs)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result
```

`_parse_args` 方法负责将字符串或字典形式的输入解析为经过 Pydantic 验证的参数：

```python
def _parse_args(self, raw_args):
    if isinstance(raw_args, str):
        raw_args = json.loads(raw_args)
    validated_args = self.args_schema.model_validate(raw_args)
    return validated_args.model_dump()
```

### 16.4.3 使用量同步机制

`_increment_usage_count` 方法中有一个关键的同步逻辑：

```python
def _increment_usage_count(self):
    self.current_usage_count += 1
    if self._original_tool is not None:
        self._original_tool.current_usage_count = self.current_usage_count
```

当 BaseTool 通过 `to_structured_tool()` 转换为 CrewStructuredTool 时，`_original_tool` 指向原始的 BaseTool 实例。每次使用计数递增时，会同步更新原始工具的计数，确保使用限制在两个对象间保持一致。

## 16.5 ToolCalling：工具调用的数据模型

`tool_calling.py` 定义了两个简洁但重要的模型：

```python
class ToolCalling(BaseModel):
    tool_name: str = Field(..., description="The name of the tool to be called.")
    arguments: dict[str, Any] | None = Field(
        ..., description="A dictionary of arguments to be passed to the tool."
    )

class InstructorToolCalling(PydanticBaseModel):
    tool_name: str = PydanticField(...)
    arguments: dict[str, Any] | None = PydanticField(...)
```

`ToolCalling` 用于普通的文本解析场景，`InstructorToolCalling` 用于支持 function calling 的 LLM。两者结构相同，但基类不同，这是为了适配不同的序列化和验证需求。

## 16.6 ToolUsage：工具调用的编排引擎

`ToolUsage` 是整个工具框架中最复杂的组件，负责从 LLM 输出的文本中解析工具调用意图，选择正确的工具，执行调用，处理错误和重试。

### 16.6.1 初始化与配置

```python
class ToolUsage:
    def __init__(self, tools_handler, tools, task, function_calling_llm,
                 agent=None, action=None, fingerprint_context=None):
        self._run_attempts = 1
        self._max_parsing_attempts = 3
        self._remember_format_after_usages = 3
        self.tools_description = render_text_description_and_args(tools)
        self.tools_names = get_tool_names(tools)
        # ...

        # 大模型优化：减少解析重试次数
        if self.function_calling_llm.model in OPENAI_BIGGER_MODELS:
            self._max_parsing_attempts = 2
            self._remember_format_after_usages = 4
```

针对 GPT-4 等大模型，框架减少了最大解析尝试次数（从 3 到 2），因为这些模型通常能在更少的尝试中生成正确格式。

### 16.6.2 工具解析流程

`_tool_calling` 方法实现了一个多层 fallback 策略：

```python
def _tool_calling(self, tool_string):
    try:
        try:
            # 第一层：直接解析 Agent 输出
            return self._original_tool_calling(tool_string, raise_error=True)
        except Exception:
            if self.function_calling_llm:
                # 第二层：使用 LLM 辅助解析
                return self._function_calling(tool_string)
            # 第三层：降级回直接解析（不抛异常）
            return self._original_tool_calling(tool_string)
    except Exception as e:
        self._run_attempts += 1
        if self._run_attempts > self._max_parsing_attempts:
            return ToolUsageError(...)
        return self._tool_calling(tool_string)  # 递归重试
```

`_validate_tool_input` 方法展示了极其健壮的输入解析策略，依次尝试四种解析方式：

1. **标准 JSON 解析** (`json.loads`)
2. **Python 字面量解析** (`ast.literal_eval`)
3. **JSON5 解析** (`json5.loads`) —— 支持注释和尾逗号
4. **JSON 修复** (`repair_json`) —— 尝试修复格式错误的 JSON

这种多重 fallback 策略是必要的，因为 LLM 生成的工具调用格式经常不是严格的 JSON。

### 16.6.3 工具选择的模糊匹配

`_select_tool` 方法使用 `SequenceMatcher` 进行模糊匹配：

```python
def _select_tool(self, tool_name):
    sanitized_input = sanitize_tool_name(tool_name)
    order_tools = sorted(
        self.tools,
        key=lambda tool: SequenceMatcher(
            None, sanitize_tool_name(tool.name), sanitized_input
        ).ratio(),
        reverse=True,
    )
    for tool in order_tools:
        sanitized_tool = sanitize_tool_name(tool.name)
        if (sanitized_tool == sanitized_input
            or SequenceMatcher(None, sanitized_tool, sanitized_input).ratio() > 0.85):
            return tool
    # 没有匹配的工具时抛出异常
    raise Exception(f"Action '{tool_name}' don't exist...")
```

这里有两个要点：首先按相似度排序所有工具，然后只接受相似度超过 0.85 的匹配。这既能容忍 LLM 输出中的小拼写错误，又不会错误匹配完全不同的工具。

### 16.6.4 执行流程与事件系统

`_use` 方法是同步执行的核心（`_ause` 是其异步版本），其执行流程包括：

1. **重复调用检测**：通过 `_check_tool_repeated_usage` 检查是否与上一次调用完全相同
2. **发出 ToolUsageStartedEvent**
3. **查询缓存**：从 `tools_handler.cache` 中读取缓存结果
4. **使用量检查**：通过 `_check_usage_limit` 确认工具未超出限制
5. **执行调用**：过滤参数、添加 fingerprint 元数据、调用 `tool.invoke`
6. **缓存写入**：通过 `cache_function` 判断是否缓存结果
7. **记录遥测数据**
8. **发出 ToolUsageFinishedEvent**

错误处理也非常完善：

```python
except Exception as e:
    self.on_tool_error(tool=tool, tool_calling=calling, e=e)
    self._run_attempts += 1
    if self._run_attempts > self._max_parsing_attempts:
        # 超过重试次数，返回错误消息
        result = ToolUsageError(f"\n{error_message}...").message
    else:
        # 还可以重试
        should_retry = True
```

### 16.6.5 Fingerprint 安全机制

`_add_fingerprint_metadata` 方法向工具参数中注入安全上下文：

```python
def _add_fingerprint_metadata(self, arguments):
    arguments = arguments.copy()
    if "security_context" not in arguments:
        arguments["security_context"] = {}

    security_context = arguments["security_context"]

    if self.agent and hasattr(self.agent, "security_config"):
        security_config = self.agent.security_config
        if security_config and hasattr(security_config, "fingerprint"):
            security_context["agent_fingerprint"] = (
                security_config.fingerprint.to_dict()
            )
    # ... 类似处理 task fingerprint
    return arguments
```

这允许工具在执行时验证调用者的身份，增加了一层安全保障。

## 16.7 CacheTools：缓存查询工具

`CacheTools` 提供了一个特殊的工具，允许 Agent 主动查询缓存：

```python
class CacheTools(BaseModel):
    name: str = "Hit Cache"
    cache_handler: CacheHandler = Field(default_factory=CacheHandler)

    def tool(self) -> CrewStructuredTool:
        return CrewStructuredTool.from_function(
            func=self.hit_cache,
            name=sanitize_tool_name(self.name),
            description="Reads directly from the cache",
        )

    def hit_cache(self, key: str) -> str | None:
        split = key.split("tool:")
        tool = split[1].split("|input:")[0].strip()
        tool_input = split[1].split("|input:")[1].strip()
        return self.cache_handler.read(tool, tool_input)
```

缓存键的格式是 `tool:<tool_name>|input:<input>`，这种结构化的键设计使得 Agent 可以精确查询之前工具调用的结果。

## 16.8 Agent 委托工具

Agent 委托工具是 CrewAI 多 Agent 协作的基础设施。

### 16.8.1 BaseAgentTool

```python
class BaseAgentTool(BaseTool):
    agents: list[BaseAgent] = Field(description="List of available agents")
    i18n: I18N = Field(default_factory=get_i18n)

    def _execute(self, agent_name, task, context=None):
        sanitized_name = self.sanitize_agent_name(agent_name)

        agent = [a for a in self.agents
                 if self.sanitize_agent_name(a.role) == sanitized_name]

        if not agent:
            return self.i18n.errors("agent_tool_unexisting_coworker").format(...)

        selected_agent = agent[0]
        task_with_assigned_agent = Task(
            description=task,
            agent=selected_agent,
            expected_output=selected_agent.i18n.slice("manager_request"),
        )
        return selected_agent.execute_task(task_with_assigned_agent, context)
```

`sanitize_agent_name` 方法对 Agent 角色名进行标准化处理（规范空白字符、移除引号、转小写），确保匹配时不受格式差异影响。

### 16.8.2 DelegateWorkTool 与 AskQuestionTool

这两个工具结构几乎相同，区别在于语义：

```python
class DelegateWorkTool(BaseAgentTool):
    name: str = "Delegate work to coworker"
    args_schema: type[BaseModel] = DelegateWorkToolSchema  # task, context, coworker

    def _run(self, task, context, coworker=None, **kwargs):
        coworker = self._get_coworker(coworker, **kwargs)
        return self._execute(coworker, task, context)

class AskQuestionTool(BaseAgentTool):
    name: str = "Ask question to coworker"
    args_schema: type[BaseModel] = AskQuestionToolSchema  # question, context, coworker

    def _run(self, question, context, coworker=None, **kwargs):
        coworker = self._get_coworker(coworker, **kwargs)
        return self._execute(coworker, question, context)
```

`AgentTools` 管理器类负责创建这些工具并注入可用的 coworker 列表：

```python
class AgentTools:
    def tools(self) -> list[BaseTool]:
        coworkers = ", ".join([f"{agent.role}" for agent in self.agents])
        delegate_tool = DelegateWorkTool(
            agents=self.agents, i18n=self.i18n,
            description=self.i18n.tools("delegate_work").format(coworkers=coworkers),
        )
        ask_tool = AskQuestionTool(
            agents=self.agents, i18n=self.i18n,
            description=self.i18n.tools("ask_question").format(coworkers=coworkers),
        )
        return [delegate_tool, ask_tool]
```

## 16.9 Memory 工具

Memory 工具让 Agent 具备主动检索和存储记忆的能力：

```python
class RecallMemoryTool(BaseTool):
    name: str = "Search memory"
    args_schema: type[BaseModel] = RecallMemorySchema  # queries: list[str]
    memory: Any = Field(exclude=True)

    def _run(self, queries: list[str] | str, **kwargs) -> str:
        if isinstance(queries, str):
            queries = [queries]
        all_lines = []
        seen_ids = set()
        for query in queries:
            matches = self.memory.recall(query, limit=20)
            for m in matches:
                if m.record.id not in seen_ids:
                    seen_ids.add(m.record.id)
                    all_lines.append(m.format())
        if not all_lines:
            return "No relevant memories found."
        return "Found memories:\n" + "\n".join(all_lines)
```

值得注意的是 `create_memory_tools` 工厂函数会根据 Memory 的 `read_only` 属性来决定是否提供 `RememberTool`：

```python
def create_memory_tools(memory):
    tools = [RecallMemoryTool(memory=memory, description=...)]
    if not memory.read_only:
        tools.append(RememberTool(memory=memory, description=...))
    return tools
```

这种条件式工具注入确保了只读 Memory 不会暴露不可用的写入能力。

## 16.10 ToolResult 与 to_langchain

`ToolResult` 是一个简单的 dataclass，用于封装工具执行结果：

```python
@dataclass
class ToolResult:
    result: str
    result_as_answer: bool = False
```

`to_langchain` 函数将 BaseTool 列表统一转换为 CrewStructuredTool 列表：

```python
def to_langchain(tools: list[BaseTool | CrewStructuredTool]) -> list[CrewStructuredTool]:
    return [t.to_structured_tool() if isinstance(t, BaseTool) else t for t in tools]
```

`BaseTool.to_structured_tool()` 方法完成转换，并通过 `_original_tool` 建立双向引用：

```python
def to_structured_tool(self) -> CrewStructuredTool:
    self._set_args_schema()
    structured_tool = CrewStructuredTool(
        name=self.name, description=self.description,
        args_schema=self.args_schema, func=self._run,
        result_as_answer=self.result_as_answer,
        max_usage_count=self.max_usage_count,
        current_usage_count=self.current_usage_count,
    )
    structured_tool._original_tool = self
    return structured_tool
```

## 16.11 自定义工具实战

掌握了工具框架的源码后，我们来看几个创建自定义工具的实际例子。

### 方式一：继承 BaseTool

```python
from pydantic import BaseModel, Field
from crewai.tools.base_tool import BaseTool

class SearchToolSchema(BaseModel):
    query: str = Field(..., description="搜索查询词")
    max_results: int = Field(default=10, description="最大结果数")

class SearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for information"
    args_schema: type[BaseModel] = SearchToolSchema

    def _run(self, query: str, max_results: int = 10) -> str:
        # 实际的搜索逻辑
        return f"Results for: {query}"
```

### 方式二：使用 @tool 装饰器

```python
from crewai.tools.base_tool import tool

@tool("calculate")
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = eval(expression)  # 简化示例
        return str(result)
    except Exception as e:
        return f"Error: {e}"
```

### 方式三：使用 CrewStructuredTool.from_function

```python
from crewai.tools.structured_tool import CrewStructuredTool

def fetch_weather(city: str, units: str = "celsius") -> str:
    """Fetch current weather for a city."""
    return f"Weather in {city}: 22 {units}"

weather_tool = CrewStructuredTool.from_function(
    func=fetch_weather,
    name="weather",
    description="Get current weather information",
)
```

## 16.12 设计洞察

回顾整个 Tools 框架，有几个架构层面的设计洞察值得总结：

1. **双层执行抽象**：BaseTool 提供面向开发者的 API，CrewStructuredTool 提供面向框架内部的统一执行接口。这种分离使得内部执行逻辑可以独立演进。

2. **渐进式容错**：从 JSON 解析到工具选择再到执行重试，每一层都有多重 fallback 策略。这在处理 LLM 的不确定输出时至关重要。

3. **Schema 即文档**：通过自动从函数签名推断 args_schema，并将其序列化为 JSON Schema 注入 prompt，实现了工具接口的自文档化。

4. **使用量作为安全阀**：`max_usage_count` 机制防止 Agent 在循环调用中浪费资源，配合重复调用检测，形成了双重保护。

5. **事件驱动的可观测性**：ToolUsage 在工具生命周期的关键节点发出事件（Started、Finished、Error），为外部监控和日志系统提供了标准化的接入点。

## 本章要点

- **BaseTool** 是所有工具的抽象基类，通过 `_run` / `_arun` 方法定义工具逻辑，支持自动从函数签名推断 `args_schema`
- **Tool 类**和 **@tool 装饰器**提供了将普通函数快速转化为工具的便捷方式，装饰器强制要求 docstring 作为工具描述
- **CrewStructuredTool** 是框架内部的统一执行层，负责参数解析、验证、使用量控制，通过 `_original_tool` 与 BaseTool 保持使用计数同步
- **ToolCalling** 和 **InstructorToolCalling** 是工具调用的数据模型，分别适配文本解析和 function calling 场景
- **ToolUsage** 是工具调用的编排引擎，实现了多层 fallback 的工具解析（JSON -> Python literal -> JSON5 -> repair），基于 SequenceMatcher 的模糊工具选择（阈值 0.85），以及带重试的执行流程
- **CacheTools** 让 Agent 能主动查询缓存，键格式为 `tool:<name>|input:<input>`
- **Agent 委托工具**（DelegateWorkTool / AskQuestionTool）通过标准化角色名匹配实现多 Agent 协作，BaseAgentTool 封装了委托执行的通用逻辑
- **Memory 工具**通过 `read_only` 属性条件式注入，只读 Memory 不暴露写入工具
- 工具框架的核心设计原则：Schema 自文档化、渐进式容错、事件驱动可观测、使用量安全阀
