# 第 4 章　Task：任务定义与 Guardrail 机制

Task 是 CrewAI 中连接 Agent 与 Crew 的纽带。每个 Task 描述了一项需要完成的工作，包括任务内容、期望输出、负责执行的 Agent，以及对输出质量的验证机制。本章将深入分析 Task 的核心字段、条件执行、输出格式、Guardrail 机制和并行执行模型。

## 4.1 Task 核心字段

Task 类定义在 `task.py` 中，继承自 Pydantic 的 `BaseModel`。其核心字段定义如下：

```python
class Task(BaseModel):
    __hash__ = object.__hash__

    description: str = Field(description="Description of the actual task.")
    expected_output: str = Field(
        description="Clear definition of expected output for the task."
    )
    agent: BaseAgent | None = Field(default=None)
    tools: list[BaseTool] | None = Field(default_factory=list)
    context: list[Task] | None | _NotSpecified = Field(default=NOT_SPECIFIED)
    async_execution: bool | None = Field(default=False)
    output_json: type[BaseModel] | None = Field(default=None)
    output_pydantic: type[BaseModel] | None = Field(default=None)
    response_model: type[BaseModel] | None = Field(default=None)
    output_file: str | None = Field(default=None)
    callback: Any | None = Field(default=None)
    human_input: bool | None = Field(default=False)
    guardrail: GuardrailType | None = Field(default=None)
    guardrails: GuardrailsType | None = Field(default=None)
    guardrail_max_retries: int = Field(default=3)
```

Task 只有两个真正的必填字段：**description**（任务描述）和 **expected_output**（期望输出）。校验在 model_validator 中完成：

```python
@model_validator(mode="after")
def validate_required_fields(self) -> Self:
    for field in ["description", "expected_output"]:
        if getattr(self, field) is None:
            raise ValueError(f"{field} must be provided either directly or through config")
    return self
```

### context 字段的特殊设计

`context` 字段使用了一个精妙的 sentinel 模式：默认值为 `NOT_SPECIFIED` 而非 `None`。这让框架能区分"用户明确设置 context=None（不需要上下文）"和"用户未设置 context（Crew 可自动注入上下文）"两种情况。

### 输出类型互斥与工具继承

Task 只允许设置一种结构化输出格式（`output_json` 或 `output_pydantic`），且如果 Task 本身没有指定 tools，会自动继承 Agent 的工具列表：

```python
@model_validator(mode="after")
def check_output(self) -> Self:
    output_types = [self.output_json, self.output_pydantic]
    if len([type for type in output_types if type]) > 1:
        raise PydanticCustomError(
            "output_type",
            "Only one output type can be set, either output_pydantic or output_json.",
            {},
        )
    return self

@model_validator(mode="after")
def check_tools(self) -> Self:
    if not self.tools and self.agent and self.agent.tools:
        self.tools = self.agent.tools
    return self
```

## 4.2 Task 的执行流程

Task 提供同步和异步两种执行入口。

```python
def execute_sync(self, agent=None, context=None, tools=None) -> TaskOutput:
    self.start_time = datetime.datetime.now()
    return self._execute_core(agent, context, tools)

def execute_async(self, agent=None, context=None, tools=None) -> Future[TaskOutput]:
    future: Future[TaskOutput] = Future()
    threading.Thread(
        daemon=True, target=self._execute_task_async,
        args=(agent, context, tools, future),
    ).start()
    return future
```

核心逻辑在 `_execute_core` 中：

```python
def _execute_core(self, agent, context, tools) -> TaskOutput:
    task_id_token = set_current_task_id(str(self.id))
    self._store_input_files()
    try:
        agent = agent or self.agent
        if not agent:
            raise Exception(f"The task '{self.description}' has no agent assigned...")

        self.prompt_context = context
        tools = tools or self.tools or []
        self.processed_by_agents.add(agent.role)

        crewai_event_bus.emit(self, TaskStartedEvent(context=context, task=self))
        result = agent.execute_task(task=self, context=context, tools=tools)
```

调用链为：Task._execute_core → Agent.execute_task → CrewAgentExecutor.invoke。执行完成后，结果被封装为 `TaskOutput`，接着依次执行 Guardrail 验证、callback 回调和文件保存。

Task 的 `prompt()` 方法负责将 description 和 expected_output 组装成完整的 prompt，可选追加 Markdown 格式要求和 trigger payload。

## 4.3 ConditionalTask：条件任务

`ConditionalTask` 定义在 `tasks/conditional_task.py` 中，支持基于前一个任务输出的条件执行：

```python
class ConditionalTask(Task):
    condition: Callable[[TaskOutput], bool] | None = Field(
        default=None,
        description="Function that determines whether the task should be executed.",
    )

    def should_execute(self, context: TaskOutput) -> bool:
        if self.condition is None:
            raise ValueError("No condition function set for conditional task")
        return self.condition(context)

    def get_skipped_task_output(self) -> TaskOutput:
        return TaskOutput(
            description=self.description,
            raw="",
            agent=self.agent.role if self.agent else "",
            output_format=OutputFormat.RAW,
        )
```

设计要点：

- ConditionalTask 不能是 Crew 中的唯一任务，也不能是第一个任务（需要前序任务的输出）
- **condition** 函数接收前一个任务的 `TaskOutput`，返回 `bool` 决定是否执行
- 被跳过时通过 `get_skipped_task_output` 返回空的占位输出，保持任务链的连续性

使用示例：

```python
conditional_task = ConditionalTask(
    description="Send notification email",
    expected_output="Email sent confirmation",
    agent=notifier_agent,
    condition=lambda output: "critical" in output.raw.lower(),
)
```

## 4.4 TaskOutput：三种输出格式

`TaskOutput` 定义在 `tasks/task_output.py` 中，是 Task 执行结果的标准容器：

```python
class TaskOutput(BaseModel):
    description: str = Field(description="Description of the task")
    name: str | None = Field(default=None)
    expected_output: str | None = Field(default=None)
    summary: str | None = Field(default=None)
    raw: str = Field(description="Raw output of the task", default="")
    pydantic: BaseModel | None = Field(default=None)
    json_dict: dict[str, Any] | None = Field(default=None)
    agent: str = Field(description="Agent that executed the task")
    output_format: OutputFormat = Field(default=OutputFormat.RAW)
    messages: list[LLMMessage] = Field(default=[])
```

三种输出格式互补存在：

1. **RAW**：`raw` 字段，始终存在，是 LLM 返回的原始字符串
2. **Pydantic**：`pydantic` 字段，当 Task 设置了 `output_pydantic` 时自动转换
3. **JSON**：`json_dict` 字段，当 Task 设置了 `output_json` 时自动解析为字典

格式转换在 `_export_output` 中通过 `Converter` 类完成，利用 LLM 将自然语言文本转换为符合 schema 的结构化数据。

summary 字段在初始化时自动从 description 截取前 10 个词：

```python
@model_validator(mode="after")
def set_summary(self):
    excerpt = " ".join(self.description.split(" ")[:10])
    self.summary = f"{excerpt}..."
    return self
```

`__str__` 遵循结构化优先原则：优先返回 pydantic，其次 json_dict，最后 raw。

## 4.5 Guardrail 机制

Guardrail 是 CrewAI 的质量控制机制，允许对 Task 输出进行验证，不通过则自动重试。

### 类型体系

```python
# utilities/guardrail_types.py
GuardrailCallable: TypeAlias = Callable[
    [TaskOutput | LiteAgentOutput], tuple[bool, Any]
]
GuardrailType: TypeAlias = GuardrailCallable | str
GuardrailsType: TypeAlias = Sequence[GuardrailType] | GuardrailType
```

Guardrail 函数的契约：接受 `TaskOutput`，返回 `tuple[bool, Any]`——第一个元素表示是否通过，第二个元素在通过时是结果数据，失败时是错误信息。

### GuardrailResult 与 process_guardrail

```python
class GuardrailResult(BaseModel):
    success: bool
    result: Any | None = None
    error: str | None = None

    @classmethod
    def from_tuple(cls, result: tuple[bool, Any | str]) -> Self:
        success, data = result
        return cls(success=success,
                   result=data if success else None,
                   error=data if not success else None)

def process_guardrail(output, guardrail, retry_count, ...) -> GuardrailResult:
    # 发送 LLMGuardrailStartedEvent
    result = guardrail(output)
    guardrail_result = GuardrailResult.from_tuple(result)
    # 发送 LLMGuardrailCompletedEvent
    return guardrail_result
```

### 单个与多个 Guardrails

Task 支持两种配置：`guardrail`（单个）和 `guardrails`（列表）。设置 `guardrails` 时会清空 `guardrail`，避免重复执行。字符串形式的 Guardrail 会自动转换为 `LLMGuardrail` 实例。

### LLMGuardrail：用 LLM 验证输出

```python
class LLMGuardrail:
    def __init__(self, description: str, llm: BaseLLM):
        self.description = description
        self.llm = llm

    def _validate_output(self, task_output: TaskOutput) -> LiteAgentOutput:
        agent = Agent(
            role="Guardrail Agent",
            goal="Validate the output of the task",
            backstory="You are a expert at validating the output of a task.",
            llm=self.llm,
        )
        query = f"""
        Ensure the following task result complies with the given guardrail.
        Task result: {task_output.raw}
        Guardrail: {self.description}
        """
        return agent.kickoff(query, response_format=LLMGuardrailResult)

    def __call__(self, task_output: TaskOutput) -> tuple[bool, Any]:
        result = self._validate_output(task_output)
        if result.pydantic.valid:
            return True, task_output.raw
        return False, result.pydantic.feedback
```

LLMGuardrail 的精妙之处：它内部创建临时 Agent 执行验证，Guardrail 本身也是一次完整的 Agent 执行。

### HallucinationGuardrail

`tasks/hallucination_guardrail.py` 中定义了幻觉检测 Guardrail。开源版为 no-op 占位，通过 `_validate_output_hook` 全局钩子支持平台版注入真实逻辑：

```python
class HallucinationGuardrail:
    def __init__(self, llm, context=None, threshold=None, tool_response=""):
        self.context = context
        self.llm = llm
        self.threshold = threshold

    def __call__(self, task_output: TaskOutput) -> tuple[bool, Any]:
        if callable(_validate_output_hook):
            return _validate_output_hook(self, task_output)
        return True, task_output.raw  # 开源版默认通过
```

### Guardrail 重试流程

`_invoke_guardrail_function` 实现完整的重试循环：

```python
def _invoke_guardrail_function(self, task_output, agent, tools, guardrail,
                                guardrail_index=None) -> TaskOutput:
    max_attempts = self.guardrail_max_retries + 1
    for attempt in range(max_attempts):
        guardrail_result = process_guardrail(output=task_output, guardrail=guardrail, ...)
        if guardrail_result.success:
            if isinstance(guardrail_result.result, str):
                task_output.raw = guardrail_result.result
                task_output.pydantic, task_output.json_dict = self._export_output(...)
            return task_output

        if attempt >= self.guardrail_max_retries:
            raise Exception(f"Task failed guardrail validation after {self.guardrail_max_retries} retries.")

        # 将错误信息作为 context 反馈给 Agent 重新执行
        context = self.i18n.errors("validation_error").format(
            guardrail_result_error=guardrail_result.error,
            task_output=task_output.raw,
        )
        result = agent.execute_task(task=self, context=context, tools=tools)
        task_output = TaskOutput(raw=result, ...)
    return task_output
```

重试机制的关键设计：每个 Guardrail 有独立的重试计数器（`_guardrail_retry_counts` 字典），错误信息会作为 context 反馈给 Agent 让其修正输出。

## 4.6 async_execution 与并行任务组

`async_execution` 字段控制任务在 Crew 编排中是否允许并行执行（注意：这不是 Python 的 async/await，而是线程级并行）。

当 `async_execution=True` 时，Crew 通过 `execute_async` 启动任务，返回 `Future` 对象：

```python
def execute_async(self, agent=None, context=None, tools=None) -> Future[TaskOutput]:
    future: Future[TaskOutput] = Future()
    threading.Thread(
        daemon=True, target=self._execute_task_async,
        args=(agent, context, tools, future),
    ).start()
    return future
```

在 Crew 的 sequential process 中，连续的 `async_execution=True` 任务形成一个并行组——Crew 同时启动这些任务，在遇到下一个同步任务时等待所有 Future 完成，收集输出作为后续上下文。

此外，Task 也提供原生 async/await 接口 `aexecute_sync`，在 Crew 的 `kickoff_async()` 模式下使用。

## 4.7 输入模板与文件处理

### 模板插值

Task 支持在 description 和 expected_output 中使用 `{variable}` 模板变量。在 Crew kickoff 时，`interpolate_inputs_and_add_conversation_history` 方法完成替换：

```python
def interpolate_inputs_and_add_conversation_history(
    self, inputs: dict[str, str | int | float | dict[str, Any] | list[Any]]
) -> None:
    if self._original_description is None:
        self._original_description = self.description
    if self._original_expected_output is None:
        self._original_expected_output = self.expected_output

    self.description = interpolate_only(
        input_string=self._original_description, inputs=inputs
    )
    self.expected_output = interpolate_only(
        input_string=self._original_expected_output, inputs=inputs
    )
```

框架保存原始文本的私有副本（`_original_description`、`_original_expected_output`），使得同一个 Task 定义可以在多次 `crew.kickoff()` 中使用不同的输入值，而不会出现模板被覆盖的问题。

如果 inputs 中包含 `crew_chat_messages`，方法还会将对话历史追加到 description 中，支持多轮对话场景。

### 文件输入输出

Task 通过 `input_files` 字段接受命名文件输入：

```python
input_files: dict[str, FileInput] = Field(
    default_factory=dict,
    description="Named input files for this task.",
)
```

执行前通过 `_store_input_files` 存入文件存储，Agent 可以在 prompt 中看到文件列表并通过工具访问。

输出文件通过 `output_file` 字段指定路径，路径安全性通过多重校验保证：

```python
@field_validator("output_file")
@classmethod
def output_file_validation(cls, value: str | None) -> str | None:
    if value is None:
        return None
    if ".." in value:
        raise ValueError("Path traversal attempts are not allowed")
    if value.startswith(("~", "$")):
        raise ValueError("Shell expansion characters are not allowed")
    if any(char in value for char in ["|", ">", "<", "&", ";"]):
        raise ValueError("Shell special characters are not allowed")
    return value
```

## 4.8 Callback 与执行时间统计

Task 完成后支持触发两层回调：

```python
# Task 级回调
if self.callback:
    cb_result = self.callback(self.output)
    if inspect.iscoroutine(cb_result):
        asyncio.run(cb_result)

# Crew 级回调（如果与 Task 回调不同）
crew = self.agent.crew
if crew and crew.task_callback and crew.task_callback != self.callback:
    cb_result = crew.task_callback(self.output)
```

两者互不干扰，如果是同一个函数则只执行一次。回调也支持协程函数。

Task 自动记录执行时间，通过 `start_time` 和 `end_time` 字段计算：

```python
@property
def execution_duration(self) -> float | None:
    if not self.start_time or not self.end_time:
        return None
    return (self.end_time - self.start_time).total_seconds()
```

`key` 属性基于 description 和 expected_output 生成 MD5 哈希值，用于唯一标识一个 Task 定义（忽略运行时状态），在训练数据和缓存场景中使用。

## 本章要点

- Task 的两个必填字段是 **description** 和 **expected_output**，它们构成传递给 Agent 的核心 prompt
- **context** 字段使用 sentinel 值 `NOT_SPECIFIED` 区分"未设置"和"显式设为 None"，支持 Crew 自动注入上下文
- **ConditionalTask** 通过 `condition` 函数实现条件执行，`should_execute` 返回 `False` 时生成空的占位输出
- **TaskOutput** 支持三种格式：raw（原始文本）、pydantic（Pydantic 模型）、json_dict（字典），类型互斥
- **Guardrail** 基于 `tuple[bool, Any]` 返回值协议，支持函数式和字符串式两种定义
- **LLMGuardrail** 内部创建临时 Agent 执行验证；**HallucinationGuardrail** 开源版为 no-op，通过钩子支持平台扩展
- Guardrail 重试最多 **guardrail_max_retries** 次（默认 3），错误信息作为 context 反馈给 Agent
- **async_execution** 通过 `threading.Thread` + `Future` 实现并行任务组，连续异步任务形成并行组
- Task 支持 **输入模板插值** 和安全的 **文件输入/输出** 机制
