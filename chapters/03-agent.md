# 第 3 章　Agent：角色设定与执行逻辑

在 CrewAI 框架中，Agent 是最核心的抽象之一。每个 Agent 都是一个拥有明确身份、目标和行为能力的智能实体。本章将深入剖析 Agent 的源码实现，从类继承体系、三要素模型、执行流程，到工具准备、知识注入和推理模式，全面揭示 Agent 的内部运作机制。

## 3.1 三要素：role / goal / backstory

CrewAI 将 Agent 的身份抽象为三个核心字段，这三个字段构成了 Agent 的"人设"，直接影响 LLM 生成的 system prompt。

在 `agents/agent_builder/base_agent.py` 中，BaseAgent 对这三个字段的定义如下：

```python
class BaseAgent(BaseModel, ABC, metaclass=AgentMeta):
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Objective of the agent")
    backstory: str = Field(description="Backstory of the agent")
```

三要素的设计哲学非常清晰：

- **role**：定义 Agent 扮演的角色，如 "Senior Data Analyst"、"Marketing Strategist"。这个字段会被注入到 system prompt 中，引导 LLM 以对应角色的口吻和视角回答问题。
- **goal**：定义 Agent 的目标，描述它需要达成什么。这让 LLM 保持对最终目标的聚焦，避免在工具调用或多轮对话中偏离方向。
- **backstory**：为 Agent 提供背景故事，赋予它经验和专业知识的暗示。例如 "You have 10 years of experience in financial modeling"，这种上下文能显著提升 LLM 的输出质量。

在 model_validator 中，框架会严格校验这三个字段必须存在：

```python
@model_validator(mode="after")
def validate_and_set_attributes(self) -> Self:
    for field in ["role", "goal", "backstory"]:
        if getattr(self, field) is None:
            raise ValueError(
                f"{field} must be provided either directly or through config"
            )
    return self
```

此外，BaseAgent 保存了原始值的私有副本（`_original_role`、`_original_goal`、`_original_backstory`），用于在 `interpolate_inputs` 时基于模板变量动态替换内容，而不会丢失原始定义。

## 3.2 继承体系：BaseAgent → Agent

### BaseAgent：抽象基类

`BaseAgent` 定义在 `agents/agent_builder/base_agent.py` 中，继承自 `BaseModel` 和 `ABC`，使用自定义的 `AgentMeta` 元类。它是所有 Agent 的抽象根类，定义了公共字段和抽象接口：

```python
class BaseAgent(BaseModel, ABC, metaclass=AgentMeta):
    # 核心字段
    role: str
    goal: str
    backstory: str
    cache: bool = Field(default=True)
    verbose: bool = Field(default=False)
    max_rpm: int | None = Field(default=None)
    allow_delegation: bool = Field(default=False)
    tools: list[BaseTool] | None = Field(default_factory=list)
    max_iter: int = Field(default=25)
    agent_executor: Any = Field(default=None)
    llm: Any = Field(default=None)
    crew: Any = Field(default=None)
    knowledge: Knowledge | None = Field(default=None)
    knowledge_sources: list[BaseKnowledgeSource] | None = Field(default=None)
    memory: Any = Field(default=None)
    apps: list[PlatformAppOrAction] | None = Field(default=None)
    mcps: list[str | MCPServerConfig] | None = Field(default=None)
```

BaseAgent 声明了若干抽象方法，子类必须实现：

- `execute_task(task, context, tools)` —— 执行任务的核心方法
- `create_agent_executor(tools)` —— 创建执行器
- `get_delegation_tools(agents)` —— 获取任务委派工具
- `get_output_converter(llm, text, model, instructions)` —— 获取输出转换器

BaseAgent 还内置了工具校验逻辑。`validate_tools` 方法确保每个 tool 要么是 `BaseTool` 实例，要么具备 `name`、`func`、`description` 三个属性（兼容 LangChain 工具）：

```python
@field_validator("tools")
@classmethod
def validate_tools(cls, tools: list[Any]) -> list[BaseTool]:
    processed_tools = []
    for tool in tools:
        if isinstance(tool, BaseTool):
            processed_tools.append(tool)
        elif all(hasattr(tool, attr) for attr in ["name", "func", "description"]):
            processed_tools.append(Tool.from_langchain(tool))
        else:
            raise ValueError(f"Invalid tool type: {type(tool)}.")
    return processed_tools
```

### Agent：完整实现

`Agent` 类定义在 `agent/core.py` 中，继承自 `BaseAgent`，是面向用户的主要入口。它在 BaseAgent 基础上增加了大量字段和能力：

```python
class Agent(BaseAgent):
    max_execution_time: int | None = Field(default=None)
    step_callback: Any | None = Field(default=None)
    use_system_prompt: bool | None = Field(default=True)
    llm: str | InstanceOf[BaseLLM] | Any = Field(default=None)
    function_calling_llm: str | InstanceOf[BaseLLM] | Any | None = Field(default=None)
    allow_code_execution: bool | None = Field(default=False)
    respect_context_window: bool = Field(default=True)
    max_retry_limit: int = Field(default=2)
    reasoning: bool = Field(default=False)
    max_reasoning_attempts: int | None = Field(default=None)
    guardrail: GuardrailType | None = Field(default=None)
    guardrail_max_retries: int = Field(default=3)
    executor_class: type[CrewAgentExecutor] | type[AgentExecutor] = Field(
        default=CrewAgentExecutor
    )
```

在初始化阶段（`post_init_setup`），Agent 会自动完成 LLM 的创建和执行器的初始化：

```python
@model_validator(mode="after")
def post_init_setup(self) -> Self:
    self.llm = create_llm(self.llm)
    if self.function_calling_llm and not isinstance(
        self.function_calling_llm, BaseLLM
    ):
        self.function_calling_llm = create_llm(self.function_calling_llm)
    if not self.agent_executor:
        self._setup_agent_executor()
    if self.allow_code_execution:
        self._validate_docker_installation()
    return self
```

`create_llm` 函数会将字符串形式的 LLM 标识（如 `"gpt-4o"`）转换为 `BaseLLM` 实例，使得用户可以用极简的方式指定模型。

### LiteAgent：已废弃的轻量替代

`LiteAgent` 定义在 `lite_agent.py` 中，是一个不继承 `BaseAgent` 的独立类，直接继承 `BaseModel`。它被设计为无需 Crew 即可独立运行的轻量 Agent：

```python
class LiteAgent(FlowTrackable, BaseModel):
    """
    .. deprecated::
        LiteAgent is deprecated and will be removed in a future version.
        Use Agent().kickoff(messages) instead.
    """
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Goal of the agent")
    backstory: str = Field(description="Backstory of the agent")
    llm: str | InstanceOf[BaseLLM] | Any | None = Field(default=None)
    tools: list[BaseTool] = Field(default_factory=list)
    max_iterations: int = Field(default=15)
```

LiteAgent 在初始化时会发出 `DeprecationWarning`，官方推荐使用 `Agent().kickoff(messages)` 替代。两者的关键区别在于：Agent 拥有完整的 Crew 集成、Knowledge 系统和 Memory 支持，而 LiteAgent 只是一个简单的 LLM 对话包装器。

## 3.3 执行流程：从 execute_task 到 CrewAgentExecutor

Agent 的执行流程是理解整个框架的关键。以下是 `execute_task` 方法的完整调用链。

### 入口：execute_task

```python
def execute_task(
    self,
    task: Task,
    context: str | None = None,
    tools: list[BaseTool] | None = None,
) -> Any:
```

这是 Crew 调用 Agent 的标准入口。方法的执行步骤可以概括为以下流水线：

1. **Reasoning 推理**（若启用） → `handle_reasoning(self, task)`
2. **日期注入**（若启用） → `self._inject_date_to_task(task)`
3. **构建 prompt** → `task.prompt()` + schema 拼接 + context 格式化
4. **Memory 检索** → 通过 unified_memory 查询相关记忆
5. **Knowledge 检索** → 查询 Agent 和 Crew 级别的知识库
6. **工具准备** → `prepare_tools(self, tools, task)`
7. **Training data 注入** → `apply_training_data(self, task_prompt)`
8. **执行** → 根据是否设置超时走不同路径
9. **后处理** → 处理 tool results、发送事件、保存消息

核心执行逻辑在步骤 8 中：

```python
validate_max_execution_time(self.max_execution_time)
if self.max_execution_time is not None:
    result = self._execute_with_timeout(task_prompt, task, self.max_execution_time)
else:
    result = self._execute_without_timeout(task_prompt, task)
```

### 无超时执行

```python
def _execute_without_timeout(self, task_prompt: str, task: Task) -> Any:
    if not self.agent_executor:
        raise RuntimeError("Agent executor is not initialized.")
    return self.agent_executor.invoke(
        {
            "input": task_prompt,
            "tool_names": self.agent_executor.tools_names,
            "tools": self.agent_executor.tools_description,
            "ask_for_human_input": task.human_input,
        }
    )["output"]
```

执行器的 `invoke` 方法接收一个字典，包含 prompt 文本、可用工具名称和描述，以及是否需要人工审核的标志。

### 有超时执行

超时控制通过 `concurrent.futures.ThreadPoolExecutor` 实现：

```python
def _execute_with_timeout(self, task_prompt: str, task: Task, timeout: int) -> Any:
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            self._execute_without_timeout, task_prompt=task_prompt, task=task
        )
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            raise TimeoutError(
                f"Task '{task.description}' execution timed out after {timeout} seconds."
            ) from e
```

### 重试机制

`execute_task` 内置了失败重试机制。通过 `_times_executed` 计数器和 `max_retry_limit`（默认为 2）控制：

```python
except Exception as e:
    if isinstance(e, _passthrough_exceptions):
        raise
    self._times_executed += 1
    if self._times_executed > self.max_retry_limit:
        raise e
    result = self.execute_task(task, context, tools)  # 递归重试
```

### CrewAgentExecutor 的创建

`create_agent_executor` 是连接 Agent 配置与执行器实例的桥梁：

```python
def create_agent_executor(
    self, tools: list[BaseTool] | None = None, task: Task | None = None
) -> None:
    raw_tools: list[BaseTool] = tools or self.tools or []
    parsed_tools = parse_tools(raw_tools)
    use_native_tool_calling = self._supports_native_tool_calling(raw_tools)

    prompt = Prompts(
        agent=self,
        has_tools=len(raw_tools) > 0,
        use_native_tool_calling=use_native_tool_calling,
        i18n=self.i18n,
        use_system_prompt=self.use_system_prompt,
    ).task_execution()

    self.agent_executor = self.executor_class(
        llm=cast(BaseLLM, self.llm),
        task=task,
        agent=self,
        crew=self.crew,
        tools=parsed_tools,
        prompt=prompt,
        original_tools=raw_tools,
        max_iter=self.max_iter,
        step_callback=self.step_callback,
        function_calling_llm=self.function_calling_llm,
        respect_context_window=self.respect_context_window,
        callbacks=[TokenCalcHandler(self._token_process)],
        response_model=(
            task.response_model or task.output_pydantic or task.output_json
        ) if task else None,
    )
```

这里有一个值得注意的设计：如果 `agent_executor` 已经存在，框架不会重新创建，而是调用 `_update_executor_parameters` 更新参数，避免了不必要的初始化开销。

## 3.4 工具准备：prepare_tools 与 Knowledge 注入

### 工具准备流程

`prepare_tools` 函数定义在 `agent/utils.py` 中，逻辑简洁但至关重要：

```python
def prepare_tools(
    agent: Agent, tools: list[BaseTool] | None, task: Task
) -> list[BaseTool]:
    final_tools = tools or agent.tools or []
    agent.create_agent_executor(tools=final_tools, task=task)
    return final_tools
```

工具的优先级是：任务级工具 > Agent 级工具 > 空列表。每次执行新任务时，都会重新调用 `create_agent_executor`，确保执行器的工具列表与当前任务匹配。

Agent 支持多种工具来源：

- **直接指定**：通过 `tools` 参数传入 `BaseTool` 列表
- **Platform Apps**：通过 `apps` 字段指定企业应用（如 `"gmail"`、`"slack"`），由 `get_platform_tools` 转换
- **MCP 服务器**：通过 `mcps` 字段接入 MCP 工具服务器，由 `MCPToolResolver` 解析
- **Code Execution**：当 `allow_code_execution=True` 时，自动注入 `CodeInterpreterTool`
- **Delegation**：当 `allow_delegation=True` 时，注入 `AgentTools` 用于跨 Agent 委派

### Knowledge 注入

Knowledge 注入发生在 `execute_task` 的中间阶段，通过 `handle_knowledge_retrieval` 完成：

```python
def handle_knowledge_retrieval(
    agent: Agent, task: Task, task_prompt: str,
    knowledge_config: dict[str, Any],
    query_func: Any, crew_query_func: Any,
) -> str:
    if not (agent.knowledge or (agent.crew and agent.crew.knowledge)):
        return task_prompt

    agent.knowledge_search_query = agent._get_knowledge_search_query(task_prompt, task)
    if agent.knowledge_search_query:
        if agent.knowledge:
            agent_knowledge_snippets = query_func(
                [agent.knowledge_search_query], **knowledge_config
            )
            if agent_knowledge_snippets:
                agent.agent_knowledge_context = extract_knowledge_context(
                    agent_knowledge_snippets
                )
                task_prompt += agent.agent_knowledge_context
    return task_prompt
```

Knowledge 注入的流程是：先用 LLM 生成一个搜索查询（`_get_knowledge_search_query`），然后在知识库中检索相关片段，最后将检索结果追加到 task prompt 中。这是一种典型的 RAG（Retrieval Augmented Generation）模式。

## 3.5 Reasoning 推理模式

当 Agent 的 `reasoning` 字段设置为 `True` 时，在任务执行前会启动推理阶段。`handle_reasoning` 函数在 `agent/utils.py` 中定义：

```python
def handle_reasoning(agent: Agent, task: Task) -> None:
    if not agent.reasoning:
        return
    try:
        from crewai.utilities.reasoning_handler import (
            AgentReasoning, AgentReasoningOutput,
        )
        reasoning_handler = AgentReasoning(task=task, agent=agent)
        reasoning_output: AgentReasoningOutput = (
            reasoning_handler.handle_agent_reasoning()
        )
        task.description += f"\n\nReasoning Plan:\n{reasoning_output.plan.plan}"
    except Exception as e:
        agent._logger.log("error", f"Error during reasoning process: {e!s}")
```

推理系统的核心类定义在 `utilities/reasoning_handler.py` 中：

```python
class ReasoningPlan(BaseModel):
    plan: str = Field(description="The detailed reasoning plan for the task.")
    ready: bool = Field(description="Whether the agent is ready to execute the task.")

class AgentReasoningOutput(BaseModel):
    plan: ReasoningPlan = Field(description="The reasoning plan for the task.")

class AgentReasoning:
    def __init__(self, task: Task, agent: Agent) -> None:
        self.task = task
        self.agent = agent
        self.llm = cast(LLM, agent.llm)

    def handle_agent_reasoning(self) -> AgentReasoningOutput:
        plan, ready = self.__create_initial_plan()
        plan, ready = self.__refine_plan_if_needed(plan, ready)
        return AgentReasoningOutput(plan=ReasoningPlan(plan=plan, ready=ready))
```

推理过程分为两步：

1. **创建初始计划**（`__create_initial_plan`）：向 LLM 发送推理 prompt，要求生成一个结构化的执行计划。如果 LLM 支持 function calling，会使用 `create_reasoning_plan` 函数 schema 获取结构化输出。
2. **迭代精炼**（`__refine_plan_if_needed`）：如果初始计划的 `ready` 为 `False`，则进入迭代循环，不断精炼计划，直到 `ready` 为 `True` 或达到 `max_reasoning_attempts` 上限。

最终，推理计划会被追加到任务描述中，作为 Agent 执行任务时的"思维链"参考。

## 3.6 独立执行：kickoff 方法

除了通过 Crew 调度，Agent 也可以独立执行任务，入口是 `kickoff` 方法：

```python
def kickoff(
    self,
    messages: str | list[LLMMessage],
    response_format: type[Any] | None = None,
    input_files: dict[str, FileInput] | None = None,
) -> LiteAgentOutput | Coroutine[Any, Any, LiteAgentOutput]:
```

`kickoff` 内部调用 `_prepare_kickoff` 完成准备工作（工具解析、prompt 构建、Memory 注入），然后创建一个 `AgentExecutor` 实例执行任务。这提供了一种轻量级的使用方式，适用于不需要 Crew 编排的简单场景。

值得注意的是，`kickoff` 包含了自动异步检测逻辑：

```python
if is_inside_event_loop():
    return self.kickoff_async(messages, response_format, input_files)
```

当在 Flow 或其他异步上下文中被调用时，会自动切换到异步执行路径，用户无需显式处理 async/await。

## 3.7 最大执行时间、Cache 与 Callbacks

### 最大执行时间

`max_execution_time` 字段以秒为单位限制单次任务的执行时间。校验逻辑确保值必须为正整数：

```python
def validate_max_execution_time(max_execution_time: int | None) -> None:
    if max_execution_time is not None:
        if not isinstance(max_execution_time, int) or max_execution_time <= 0:
            raise ValueError(
                "Max Execution time must be a positive integer greater than zero"
            )
```

超时后会抛出 `TimeoutError`，附带详细的诊断信息。

### Cache 机制

BaseAgent 的 `cache` 字段默认为 `True`，配合 `CacheHandler` 实例实现工具调用结果的缓存：

```python
cache: bool = Field(
    default=True,
    description="Whether the agent should use a cache for tool usage."
)
cache_handler: CacheHandler | None = Field(default=None)
```

Cache 策略是：对于相同的工具调用和相同的参数，直接返回缓存结果，避免重复的 API 调用或计算。

### Callbacks

Agent 支持两层 callback：

- **step_callback**：每一步执行后的回调，可用于监控和日志记录
- **callbacks**：注册在 Agent 上的通用回调列表，默认包含 `TokenCalcHandler` 用于 token 用量统计

```python
step_callback: Any | None = Field(
    default=None,
    description="Callback to be executed after each step of the agent execution.",
)
callbacks: list[Callable[[Any], Any]] = Field(
    default_factory=list,
    description="Callbacks to be used for the agent"
)
```

## 3.8 RPM 控制与其他保护机制

Agent 提供了请求频率控制（`max_rpm`）和上下文窗口保护（`respect_context_window`）。

RPM 控制在 BaseAgent 初始化时设置：

```python
if self.max_rpm and not self._rpm_controller:
    self._rpm_controller = RPMController(
        max_rpm=self.max_rpm, logger=self._logger
    )
```

上下文窗口保护默认开启，当消息超出 LLM 的 token 限制时会自动进行摘要压缩：

```python
respect_context_window: bool = Field(
    default=True,
    description="Keep messages under the context window size by summarizing content.",
)
```

## 本章要点

- Agent 的身份由三要素定义：**role**（角色）、**goal**（目标）、**backstory**（背景），这三个字段共同构成 LLM system prompt 的核心内容
- 类继承体系为 **BaseAgent（抽象基类）→ Agent（完整实现）**；LiteAgent 已废弃，推荐使用 `Agent().kickoff()`
- 执行流程遵循 `execute_task → handle_reasoning → 构建 prompt → Memory/Knowledge 检索 → prepare_tools → CrewAgentExecutor.invoke` 的流水线
- **工具准备**支持多种来源：直接指定、Platform Apps、MCP 服务器、Code Execution 和 Delegation
- **Knowledge 注入**采用 RAG 模式：先用 LLM 生成搜索查询，再检索知识库片段追加到 prompt
- **Reasoning 推理模式**在任务执行前生成结构化的执行计划，通过迭代精炼确保计划质量
- **超时控制**通过 ThreadPoolExecutor 实现，**失败重试**通过递归调用和计数器控制
- **Cache** 默认开启以避免重复工具调用，**RPM 控制**防止 API 限流，**respect_context_window** 自动摘要防止 token 溢出
- Agent 支持 **kickoff** 独立执行模式，可自动检测异步上下文并切换执行路径
