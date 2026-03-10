# 第 12 章　LLM Hooks 与 Tool Hooks

在生产环境中运行 AI Agent 系统时，我们需要监控每一次 LLM 调用、审计每一次工具执行、必要时修改输入输出甚至阻止某些操作。CrewAI 的 Hooks 系统正是为此而设计——它提供了一套基于 decorator 模式的 Hook 注册机制，允许用户在 LLM 调用和工具调用的前后插入自定义逻辑。

## 12.1 Hooks 系统总览

Hooks 系统的代码组织在 `hooks/` 目录下：

```
crewai/hooks/
├── __init__.py         # 统一导出接口
├── decorators.py       # @before_llm_call / @after_llm_call 等装饰器
├── llm_hooks.py        # LLMCallHookContext + LLM Hook 注册表
├── tool_hooks.py       # ToolCallHookContext + Tool Hook 注册表
├── types.py            # Hook Protocol 类型定义
└── wrappers.py         # HookMethod 包装器（用于 @CrewBase 类方法）
```

系统围绕四个核心 Hook 点展开：

| Hook 点 | 时机 | 返回值语义 |
|---------|------|-----------|
| `before_llm_call` | LLM 调用之前 | `False` 阻止调用，`None/True` 允许 |
| `after_llm_call` | LLM 调用之后 | `str` 替换响应，`None` 保持原响应 |
| `before_tool_call` | 工具调用之前 | `False` 阻止调用，`None/True` 允许 |
| `after_tool_call` | 工具调用之后 | `str` 替换结果，`None` 保持原结果 |

## 12.2 Hook Protocol 类型定义

`hooks/types.py` 使用 Python 的 `Protocol` 和泛型为 Hook 系统定义了严格的类型约束：

```python
@runtime_checkable
class Hook(Protocol, Generic[ContextT, ReturnT]):
    """所有 Hook 类型的通用协议"""
    def __call__(self, context: ContextT) -> ReturnT: ...

class BeforeLLMCallHook(Hook["LLMCallHookContext", bool | None], Protocol):
    def __call__(self, context: LLMCallHookContext) -> bool | None: ...

class AfterLLMCallHook(Hook["LLMCallHookContext", str | None], Protocol):
    def __call__(self, context: LLMCallHookContext) -> str | None: ...

class BeforeToolCallHook(Hook["ToolCallHookContext", bool | None], Protocol):
    def __call__(self, context: ToolCallHookContext) -> bool | None: ...

class AfterToolCallHook(Hook["ToolCallHookContext", str | None], Protocol):
    def __call__(self, context: ToolCallHookContext) -> str | None: ...
```

设计上遵循统一的约定：

- **所有 before hooks**：返回 `bool | None`，其中 `False` 表示阻止执行
- **所有 after hooks**：返回 `str | None`，其中 `str` 表示替换原始结果

同时提供了基于 `Callable` 的类型别名以兼容函数式风格：

```python
BeforeLLMCallHookCallable = Callable[["LLMCallHookContext"], bool | None]
AfterLLMCallHookCallable = Callable[["LLMCallHookContext"], str | None]
BeforeToolCallHookCallable = Callable[["ToolCallHookContext"], bool | None]
AfterToolCallHookCallable = Callable[["ToolCallHookContext"], str | None]
```

## 12.3 LLMCallHookContext：LLM 调用上下文

`LLMCallHookContext` 是传递给 LLM Hook 函数的上下文对象，封装了调用时的全部状态信息：

```python
class LLMCallHookContext:
    executor: CrewAgentExecutor | AgentExecutor | LiteAgent | None
    messages: list[LLMMessage]
    agent: Any
    task: Any
    crew: Any
    llm: BaseLLM | None | str | Any
    iterations: int
    response: str | None  # 仅 after_llm_call 时有值

    def __init__(
        self,
        executor=None,
        response=None,
        messages=None,
        llm=None,
        agent=None,
        task=None,
        crew=None,
    ) -> None:
        if executor is not None:
            # 从 executor 中提取状态
            self.executor = executor
            self.messages = executor.messages
            self.llm = executor.llm
            self.iterations = executor.iterations
            if hasattr(executor, "agent"):
                self.agent = executor.agent
                self.task = cast("CrewAgentExecutor", executor).task
                self.crew = cast("CrewAgentExecutor", executor).crew
            else:
                # LiteAgent 场景
                self.agent = executor.original_agent if hasattr(executor, "original_agent") else executor
                self.task = None
                self.crew = None
        else:
            # 直接 LLM 调用（无 Agent 上下文）
            self.executor = None
            self.messages = messages or []
            self.llm = llm
            self.agent = agent
            self.task = task
            self.crew = crew
            self.iterations = 0
        self.response = response
```

关键设计决策：

1. **双路径初始化**：同时支持从 `executor` 自动提取状态，和直接传入参数的手动构造
2. **messages 是可变引用**：Hook 可以直接修改 `context.messages`（追加、删除元素），但**不能替换整个列表**（会断开引用）
3. **iterations 追踪**：记录当前是 Agent 的第几次迭代，可用于防止无限循环

### 人机交互支持

`LLMCallHookContext` 还提供了请求人工输入的能力：

```python
def request_human_input(
    self,
    prompt: str,
    default_message: str = "Press Enter to continue, or provide feedback:",
) -> str:
    printer = Printer()
    event_listener.formatter.pause_live_updates()
    try:
        printer.print(content=f"\n{prompt}", color="bold_yellow")
        printer.print(content=default_message, color="cyan")
        response = input().strip()
        return response
    finally:
        event_listener.formatter.resume_live_updates()
```

这个方法会暂停 Rich 的实时更新，显示提示信息，等待用户输入，然后恢复实时更新。它是 Human-in-the-Loop 工作流的关键基础设施。

## 12.4 ToolCallHookContext：工具调用上下文

`ToolCallHookContext` 是工具 Hook 的上下文对象，结构更加简洁：

```python
class ToolCallHookContext:
    def __init__(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool: CrewStructuredTool,
        agent: Agent | BaseAgent | None = None,
        task: Task | None = None,
        crew: Crew | None = None,
        tool_result: str | None = None,  # 仅 after_tool_call 时有值
    ) -> None:
        self.tool_name = tool_name
        self.tool_input = tool_input    # 可在 before_tool_call 中原地修改
        self.tool = tool
        self.agent = agent
        self.task = task
        self.crew = crew
        self.tool_result = tool_result
```

注意 `tool_input` 是一个可变字典——`before_tool_call` Hook 可以直接修改其内容来调整工具的输入参数：

```python
@before_tool_call
def sanitize_search_query(context):
    if "query" in context.tool_input:
        context.tool_input["query"] = context.tool_input["query"].strip()
    return None  # 允许执行
```

## 12.5 Decorator 模式的 Hook 注册

### 12.5.1 装饰器工厂

`hooks/decorators.py` 使用工厂函数 `_create_hook_decorator` 消除了四个 Hook 装饰器之间的代码重复：

```python
def _create_hook_decorator(
    hook_type: str,
    register_function: Callable[..., Any],
    marker_attribute: str,
) -> Callable[..., Any]:
    def decorator_factory(
        func: Callable[..., Any] | None = None,
        *,
        tools: list[str] | None = None,
        agents: list[str] | None = None,
    ) -> Callable[..., Any]:
        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            setattr(f, marker_attribute, True)

            sig = inspect.signature(f)
            params = list(sig.parameters.keys())
            is_method = len(params) >= 2 and params[0] == "self"

            if tools:
                f._filter_tools = tools
            if agents:
                f._filter_agents = agents

            if tools or agents:
                @wraps(f)
                def filtered_hook(context: Any) -> Any:
                    if tools and hasattr(context, "tool_name"):
                        if context.tool_name not in tools:
                            return None
                    if agents and hasattr(context, "agent"):
                        if context.agent and context.agent.role not in agents:
                            return None
                    return f(context)

                if not is_method:
                    register_function(filtered_hook)
                return f

            if not is_method:
                register_function(f)
            return f

        if func is None:
            return decorator
        return decorator(func)

    return decorator_factory
```

这个工厂函数的精妙之处在于：

1. **同时支持带参数和不带参数的装饰器**：`@before_llm_call` 和 `@before_llm_call(agents=["Researcher"])` 都能工作
2. **过滤器自动注入**：传入 `tools` 或 `agents` 参数时，自动创建一个包装函数进行过滤
3. **方法 vs 函数区分**：通过检查 `self` 参数判断是否为类方法。类方法**不自动注册**（由 `@CrewBase` 在实例化时处理）

### 12.5.2 四个 Hook 装饰器

基于工厂函数，四个装饰器的实现非常简洁：

```python
def before_llm_call(func=None, *, agents=None):
    from crewai.hooks.llm_hooks import register_before_llm_call_hook
    return _create_hook_decorator(
        hook_type="llm",
        register_function=register_before_llm_call_hook,
        marker_attribute="is_before_llm_call_hook",
    )(func=func, agents=agents)

def after_llm_call(func=None, *, agents=None):
    from crewai.hooks.llm_hooks import register_after_llm_call_hook
    return _create_hook_decorator(
        hook_type="llm",
        register_function=register_after_llm_call_hook,
        marker_attribute="is_after_llm_call_hook",
    )(func=func, agents=agents)

def before_tool_call(func=None, *, tools=None, agents=None):
    from crewai.hooks.tool_hooks import register_before_tool_call_hook
    return _create_hook_decorator(
        hook_type="tool",
        register_function=register_before_tool_call_hook,
        marker_attribute="is_before_tool_call_hook",
    )(func=func, tools=tools, agents=agents)

def after_tool_call(func=None, *, tools=None, agents=None):
    from crewai.hooks.tool_hooks import register_after_tool_call_hook
    return _create_hook_decorator(
        hook_type="tool",
        register_function=register_after_tool_call_hook,
        marker_attribute="is_after_tool_call_hook",
    )(func=func, tools=tools, agents=agents)
```

注意 LLM Hook 只支持 `agents` 过滤，而 Tool Hook 同时支持 `tools` 和 `agents` 过滤。

## 12.6 Hook 注册表

### 12.6.1 全局注册表

每种 Hook 类型都有独立的全局注册表（模块级列表）：

```python
# llm_hooks.py
_before_llm_call_hooks: list[BeforeLLMCallHookType | BeforeLLMCallHookCallable] = []
_after_llm_call_hooks: list[AfterLLMCallHookType | AfterLLMCallHookCallable] = []

# tool_hooks.py
_before_tool_call_hooks: list[BeforeToolCallHookType | BeforeToolCallHookCallable] = []
_after_tool_call_hooks: list[AfterToolCallHookType | AfterToolCallHookCallable] = []
```

注册和获取通过简单的列表操作实现：

```python
def register_before_llm_call_hook(hook):
    _before_llm_call_hooks.append(hook)

def get_before_llm_call_hooks():
    return _before_llm_call_hooks.copy()  # 返回副本，防止外部修改

def unregister_before_llm_call_hook(hook) -> bool:
    try:
        _before_llm_call_hooks.remove(hook)
        return True
    except ValueError:
        return False
```

### 12.6.2 清理接口

系统提供了多层次的清理接口：

```python
# 清理单个类别
clear_before_llm_call_hooks() -> int
clear_after_llm_call_hooks() -> int
clear_before_tool_call_hooks() -> int
clear_after_tool_call_hooks() -> int

# 清理 LLM 或 Tool 的所有 hooks
clear_all_llm_call_hooks() -> tuple[int, int]
clear_all_tool_call_hooks() -> tuple[int, int]

# 清理所有 hooks
def clear_all_global_hooks() -> dict[str, tuple[int, int]]:
    llm_counts = clear_all_llm_call_hooks()
    tool_counts = clear_all_tool_call_hooks()
    return {
        "llm_hooks": llm_counts,
        "tool_hooks": tool_counts,
        "total": (llm_counts[0] + tool_counts[0], llm_counts[1] + tool_counts[1]),
    }
```

## 12.7 HookMethod 包装器

当 Hook 函数是 `@CrewBase` 装饰类的方法时，不能在模块加载阶段注册（因为 `self` 还不存在）。`wrappers.py` 定义了四个 HookMethod 包装器来解决这个问题：

```python
class BeforeLLMCallHookMethod:
    is_before_llm_call_hook: bool = True

    def __init__(
        self,
        meth: Callable[[Any, LLMCallHookContext], None],
        agents: list[str] | None = None,
    ) -> None:
        self._meth = meth
        self.agents = agents
        _copy_method_metadata(self, meth)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return self._meth(*args, **kwargs)

    def __get__(self, obj: Any, objtype=None) -> Any:
        if obj is None:
            return self
        return lambda context: self._meth(obj, context)
```

核心设计点：

1. **`is_before_llm_call_hook` 标记**：`@CrewBase` 装饰器在初始化时扫描所有属性，发现此标记则注册为 Hook
2. **Descriptor 协议**：实现 `__get__` 方法，确保通过实例访问时能正确绑定 `self`
3. **元数据复制**：`_copy_method_metadata` 保持函数名、文档字符串等元数据不丢失

四个包装器类遵循相同模式：

```python
class BeforeLLMCallHookMethod:  # agents 过滤
class AfterLLMCallHookMethod:   # agents 过滤
class BeforeToolCallHookMethod: # tools + agents 过滤
class AfterToolCallHookMethod:  # tools + agents 过滤
```

## 12.8 Hook 调用链路

### 12.8.1 BaseLLM 层面的 Hook 调用

`BaseLLM` 为原生 Provider 提供了 Hook 调用的模板方法：

```python
def _invoke_before_llm_call_hooks(
    self, messages, from_agent=None
) -> bool:
    """返回 True 表示继续执行，False 表示被 Hook 阻止"""
    if from_agent is not None:
        return True  # Agent 上下文由 executor 处理

    before_hooks = get_before_llm_call_hooks()
    if not before_hooks:
        return True

    hook_context = LLMCallHookContext(
        executor=None,
        messages=messages,
        llm=self,
        agent=None, task=None, crew=None,
    )

    for hook in before_hooks:
        result = hook(hook_context)
        if result is False:
            return False
    return True
```

注意这里有一个重要的分支：**当 `from_agent` 不为 None 时，Hook 调用跳过**。这是因为 Agent 场景下的 Hook 由 `CrewAgentExecutor` 在更高层级调用，避免重复执行。

`_invoke_after_llm_call_hooks` 的逻辑类似，但支持响应修改：

```python
def _invoke_after_llm_call_hooks(self, messages, response, from_agent=None) -> str:
    if from_agent is not None or not isinstance(response, str):
        return response

    after_hooks = get_after_llm_call_hooks()
    if not after_hooks:
        return response

    hook_context = LLMCallHookContext(
        executor=None, messages=messages, llm=self,
        response=response,
    )
    modified_response = response

    for hook in after_hooks:
        result = hook(hook_context)
        if result is not None and isinstance(result, str):
            modified_response = result
            hook_context.response = modified_response  # 级联传递

    return modified_response
```

### 12.8.2 LLM.call 中的 Hook 集成

在 LiteLLM 回退路径中，`LLM.call` 方法在关键位置调用 Hook：

```python
def call(self, messages, tools=None, ...):
    with llm_call_context() as call_id:
        # 1. 发出调用开始事件
        crewai_event_bus.emit(self, event=LLMCallStartedEvent(...))

        # 2. before hook
        if not self._invoke_before_llm_call_hooks(messages, from_agent):
            raise ValueError("LLM call blocked by before_llm_call hook")

        # 3. 执行 LLM 调用
        result = self._handle_non_streaming_response(params, ...)

        # 4. after hook
        if isinstance(result, str):
            result = self._invoke_after_llm_call_hooks(messages, result, from_agent)

        return result
```

## 12.9 使用场景与示例

### 12.9.1 监控与日志

最简单的用法是记录所有 LLM 调用：

```python
from crewai.hooks import before_llm_call, after_llm_call

@before_llm_call
def log_llm_calls(context):
    agent_name = context.agent.role if context.agent else "Direct"
    print(f"[LLM] Agent={agent_name}, Messages={len(context.messages)}, "
          f"Iteration={context.iterations}")

@after_llm_call
def log_response_length(context):
    if context.response:
        print(f"[LLM] Response length: {len(context.response)} chars")
    return None  # 不修改响应
```

### 12.9.2 输出清洗

使用 `after_llm_call` 清洗 LLM 响应中的敏感信息：

```python
@after_llm_call
def sanitize_response(context):
    if context.response and "SECRET_KEY" in context.response:
        return context.response.replace("SECRET_KEY", "[REDACTED]")
    return None  # 保持原响应
```

### 12.9.3 工具调用审批门

使用 `before_tool_call` 实现危险操作的人工审批：

```python
@before_tool_call(tools=["delete_file", "execute_code"])
def approve_dangerous_tools(context):
    response = context.request_human_input(
        prompt=f"Tool '{context.tool_name}' is about to execute with args: {context.tool_input}",
        default_message="Type 'approve' to continue, anything else to block:"
    )
    if response.lower() != "approve":
        return False  # 阻止执行
    return None  # 允许执行
```

### 12.9.4 按 Agent 过滤

只对特定 Agent 的调用应用 Hook：

```python
@before_llm_call(agents=["Researcher", "Analyst"])
def log_research_calls(context):
    print(f"Research call by {context.agent.role}")

@after_tool_call(tools=["web_search"], agents=["Researcher"])
def log_search_results(context):
    print(f"Search returned {len(context.tool_result)} chars")
    return None
```

### 12.9.5 迭代次数限制

防止 Agent 陷入无限循环：

```python
@before_llm_call
def limit_iterations(context):
    if context.iterations > 15:
        response = context.request_human_input(
            prompt=f"Agent has reached {context.iterations} iterations. Continue?",
        )
        if response.lower() == "no":
            return False
    return None
```

### 12.9.6 工具输入修改

在工具执行前修改输入参数：

```python
@before_tool_call(tools=["web_search"])
def add_search_constraints(context):
    context.tool_input["max_results"] = 5
    if "site:" not in context.tool_input.get("query", ""):
        context.tool_input["query"] += " site:docs.crewai.com"
    return None  # 允许执行
```

## 12.10 @CrewBase 中的 Hook 方法

在使用 `@CrewBase` 装饰器的类中，Hook 可以定义为实例方法，访问类的状态：

```python
from crewai import CrewBase
from crewai.hooks import before_llm_call, after_tool_call

@CrewBase
class MyResearchCrew:
    def __init__(self):
        self.call_count = 0
        self.total_chars = 0

    @before_llm_call
    def track_calls(self, context):
        self.call_count += 1
        if self.call_count > 100:
            print(f"Warning: {self.call_count} LLM calls made")

    @after_tool_call(tools=["web_search"])
    def track_search_output(self, context):
        if context.tool_result:
            self.total_chars += len(context.tool_result)
        return None
```

底层机制是：

1. `@before_llm_call` 检测到 `self` 参数，**不自动注册**到全局注册表
2. 函数被包装为 `BeforeLLMCallHookMethod` 实例
3. `@CrewBase` 在 `__init__` 时扫描属性，找到 `is_before_llm_call_hook = True` 的属性
4. 将绑定了 `self` 的方法注册为 crew-scoped 的 Hook

这种设计确保了 Hook 方法与 Crew 实例的生命周期一致。

## 12.11 Hook 与 Event Bus 的关系

Hooks 和 Event Bus 是 CrewAI 中两个不同层次的可观测性机制：

| 特性 | Hooks | Event Bus |
|------|-------|-----------|
| **目的** | 拦截并修改执行流程 | 被动观察事件 |
| **可修改性** | 可以修改输入/输出/阻止执行 | 只读观察 |
| **注册方式** | Decorator / register 函数 | @listener decorator |
| **粒度** | LLM 调用 / 工具调用 | 更细粒度（chunk、thinking 等） |
| **执行模型** | 同步链式执行 | 发布-订阅 |

两者互补而非替代：Hooks 用于主动干预执行流程，Event Bus 用于被动收集遥测数据。

## 本章要点

- CrewAI 提供四个 Hook 点：`@before_llm_call`、`@after_llm_call`、`@before_tool_call`、`@after_tool_call`
- **before hooks** 返回 `False` 可阻止执行，**after hooks** 返回 `str` 可替换输出
- `LLMCallHookContext` 封装了 executor、messages、agent、task、crew、llm 等完整上下文信息，并支持 `request_human_input()` 人机交互
- `ToolCallHookContext` 提供 tool_name、tool_input（可变）、tool_result 等工具调用状态
- 装饰器通过 `_create_hook_decorator` 工厂函数统一实现，支持带参数（`agents`/`tools` 过滤）和不带参数两种使用方式
- Hook 注册表是模块级列表，提供 register / unregister / clear 等完整的生命周期管理
- `HookMethod` 包装器通过 Descriptor 协议支持 `@CrewBase` 类中的实例方法 Hook
- Hook 在 `BaseLLM._invoke_before_llm_call_hooks` 和 `LLM.call` 中被调用，Agent 场景由 executor 层处理以避免重复
- Hooks 与 Event Bus 互补：前者用于主动干预，后者用于被动观察
