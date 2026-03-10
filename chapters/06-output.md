# 第 6 章　输出系统：CrewOutput 与结构化输出

CrewAI 的输出系统提供了从任务级到 Crew 级的完整输出抽象，支持原始文本、JSON 字典和 Pydantic 模型三种格式，并内置文件保存和 token 统计功能。本章深入分析三个核心组件：`OutputFormat`、`TaskOutput` 和 `CrewOutput`。

---

## 6.1 OutputFormat：输出格式枚举

定义在 `tasks/output_format.py`：

```python
class OutputFormat(str, Enum):
    """Enum that represents the output format of a task."""
    JSON = "json"
    PYDANTIC = "pydantic"
    RAW = "raw"
```

| 格式 | 说明 | 适用场景 |
|------|------|----------|
| `RAW` | 原始字符串 | 自由格式文本：文章、总结、创意写作 |
| `JSON` | Python 字典 | 结构化数据，不需类型安全 |
| `PYDANTIC` | Pydantic 模型实例 | 严格类型验证和 IDE 自动补全 |

继承 `str` 和 `Enum` 使其既可作枚举值，也可直接用于序列化。

---

## 6.2 TaskOutput：任务级输出

定义在 `tasks/task_output.py`：

```python
class TaskOutput(BaseModel):
    """Class that represents the result of a task."""

    description: str = Field(description="Description of the task")
    name: str | None = Field(description="Name of the task", default=None)
    expected_output: str | None = Field(
        description="Expected output of the task", default=None
    )
    summary: str | None = Field(description="Summary of the task", default=None)
    raw: str = Field(description="Raw output of the task", default="")
    pydantic: BaseModel | None = Field(
        description="Pydantic output of task", default=None
    )
    json_dict: dict[str, Any] | None = Field(
        description="JSON dictionary of task", default=None
    )
    agent: str = Field(description="Agent that executed the task")
    output_format: OutputFormat = Field(
        description="Output format of the task", default=OutputFormat.RAW
    )
    messages: list[LLMMessage] = Field(description="Messages of the task", default=[])
```

### 6.2.1 字段设计

体现"多重表示、同一结果"思路：

- **`raw`**：始终存在的原始字符串，最基础的表示
- **`pydantic`**：当 Task 设置了 `output_pydantic` 时填充
- **`json_dict`**：当 Task 设置了 `output_json` 时填充
- **`agent`**：存储 Agent 的 `role` 字符串而非对象引用，避免序列化时的循环引用
- **`messages`**：完整 LLM 对话历史，用于调试和追踪

### 6.2.2 自动摘要

```python
@model_validator(mode="after")
def set_summary(self):
    excerpt = " ".join(self.description.split(" ")[:10])
    self.summary = f"{excerpt}..."
    return self
```

从任务描述提取前 10 个单词作为摘要，在日志和执行回放中足以识别任务。

### 6.2.3 安全的 JSON 访问

```python
@property
def json(self) -> str | None:
    if self.output_format != OutputFormat.JSON:
        raise ValueError(
            "Invalid output format requested. "
            "Please make sure to set the output_json property for the task"
        )
    return json.dumps(self.json_dict)
```

若输出格式不是 JSON，直接抛出明确异常引导用户正确配置。

### 6.2.4 to_dict() 和 __str__()

```python
def to_dict(self) -> dict[str, Any]:
    output_dict = {}
    if self.json_dict:
        output_dict.update(self.json_dict)
    elif self.pydantic:
        output_dict.update(self.pydantic.model_dump())
    return output_dict

def __str__(self) -> str:
    if self.pydantic:
        return str(self.pydantic)
    if self.json_dict:
        return str(self.json_dict)
    return self.raw
```

`to_dict()` 优先使用 `json_dict`（原生字典，零转换成本），`__str__()` 实现优雅降级链，确保任何情况下都返回有意义内容。

---

## 6.3 CrewOutput：Crew 级输出

定义在 `crews/crew_output.py`：

```python
class CrewOutput(BaseModel):
    """Class that represents the result of a crew."""

    raw: str = Field(description="Raw output of crew", default="")
    pydantic: BaseModel | None = Field(
        description="Pydantic output of Crew", default=None
    )
    json_dict: dict[str, Any] | None = Field(
        description="JSON dict output of Crew", default=None
    )
    tasks_output: list[TaskOutput] = Field(
        description="Output of each task", default=[]
    )
    token_usage: UsageMetrics = Field(
        description="Processed token summary", default_factory=UsageMetrics
    )
```

### 6.3.1 构建过程

Crew 的 `_create_crew_output()` 方法从最后一个有效任务提取最终输出：

```python
def _create_crew_output(self, task_outputs: list[TaskOutput]) -> CrewOutput:
    valid_outputs = [t for t in task_outputs if t.raw]
    if not valid_outputs:
        raise ValueError("No valid task outputs available to create crew output.")
    final_task_output = valid_outputs[-1]

    return CrewOutput(
        raw=final_task_output.raw,
        pydantic=final_task_output.pydantic,
        json_dict=final_task_output.json_dict,
        tasks_output=task_outputs,
        token_usage=self.token_usage,
    )
```

关键设计：**Crew 的最终输出取自最后一个有效任务**。要让 Crew 输出结构化数据，只需在最后一个 Task 上设置 `output_pydantic` 或 `output_json`。

### 6.3.2 字典式访问

```python
def __getitem__(self, key):
    if self.pydantic and hasattr(self.pydantic, key):
        return getattr(self.pydantic, key)
    if self.json_dict and key in self.json_dict:
        return self.json_dict[key]
    raise KeyError(f"Key '{key}' not found in CrewOutput.")
```

让 `CrewOutput` 可像字典一样使用 `result["field_name"]`，统一了 Pydantic 和 JSON 两种结构化输出的访问方式。

### 6.3.3 json 属性

```python
@property
def json(self) -> str | None:
    if self.tasks_output[-1].output_format != OutputFormat.JSON:
        raise ValueError(
            "No JSON output found in the final task. Please make sure to set "
            "the output_json property in the final task in your crew."
        )
    return json.dumps(self.json_dict)
```

错误消息明确指出需要在最后一个 Task 上设置 `output_json`。

---

## 6.4 UsageMetrics：Token 统计

```python
class UsageMetrics(BaseModel):
    total_tokens: int = Field(default=0)
    prompt_tokens: int = Field(default=0)
    cached_prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    successful_requests: int = Field(default=0)

    def add_usage_metrics(self, usage_metrics: Self) -> None:
        self.total_tokens += usage_metrics.total_tokens
        self.prompt_tokens += usage_metrics.prompt_tokens
        self.cached_prompt_tokens += usage_metrics.cached_prompt_tokens
        self.completion_tokens += usage_metrics.completion_tokens
        self.successful_requests += usage_metrics.successful_requests
```

Crew 执行结束时遍历所有 Agent（含 Manager Agent）的 LLM 实例，通过累加模式汇总 token 使用量：

```python
def calculate_usage_metrics(self) -> UsageMetrics:
    total_usage_metrics = UsageMetrics()
    for agent in self.agents:
        if isinstance(agent.llm, BaseLLM):
            llm_usage = agent.llm.get_token_usage_summary()
            total_usage_metrics.add_usage_metrics(llm_usage)
    if self.manager_agent and isinstance(self.manager_agent.llm, BaseLLM):
        llm_usage = self.manager_agent.llm.get_token_usage_summary()
        total_usage_metrics.add_usage_metrics(llm_usage)
    return total_usage_metrics
```

`cached_prompt_tokens` 追踪利用 LLM provider 缓存的 prompt token 数量，对成本优化很有价值。

---

## 6.5 Task 的输出格式配置

### 6.5.1 output_pydantic

```python
output_pydantic: type[BaseModel] | None = Field(
    description="A Pydantic model to be used to create a Pydantic output.",
    default=None,
)
```

LLM 输出会被解析为指定的 Pydantic 模型实例。

### 6.5.2 output_json

```python
output_json: type[BaseModel] | None = Field(
    description="A Pydantic model to be used to create a JSON output.",
    default=None,
)
```

类型也是 `type[BaseModel]`——用 Pydantic 模型描述 JSON 结构以获得验证能力，但最终输出是 Python 字典。可复用同一模型类于两种用途。

### 6.5.3 互斥约束

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
```

### 6.5.4 格式判定

```python
def _get_output_format(self) -> OutputFormat:
    if self.output_json:
        return OutputFormat.JSON
    if self.output_pydantic:
        return OutputFormat.PYDANTIC
    return OutputFormat.RAW
```

---

## 6.6 输出转换流程

Task 的 `_execute_core()` 中处理三种情况：

```python
result = agent.execute_task(task=self, context=context, tools=tools)

if isinstance(result, BaseModel):
    # 情况 1：Agent/LLM 直接返回 Pydantic 模型（structured output 特性）
    raw = result.model_dump_json()
    if self.output_pydantic:
        pydantic_output = result
        json_output = None
    elif self.output_json:
        pydantic_output = None
        json_output = result.model_dump()
elif not self._guardrails and not self._guardrail:
    # 情况 2：返回字符串，无 guardrail，通过 _export_output 转换
    raw = result
    pydantic_output, json_output = self._export_output(result)
else:
    # 情况 3：有 guardrail，延后转换
    raw = result
    pydantic_output, json_output = None, None
```

`_export_output()` 调用 `convert_to_model()` 将文本解析为 Pydantic 模型：

```python
def _export_output(self, result: str) -> tuple[BaseModel | None, dict[str, Any] | None]:
    pydantic_output, json_output = None, None
    if self.output_pydantic or self.output_json:
        model_output = convert_to_model(
            result, self.output_pydantic, self.output_json,
            self.agent, self.converter_cls,
        )
        if isinstance(model_output, BaseModel):
            pydantic_output = model_output
        elif isinstance(model_output, dict):
            json_output = model_output
        elif isinstance(model_output, str):
            try:
                json_output = json.loads(model_output)
            except json.JSONDecodeError:
                json_output = None
    return pydantic_output, json_output
```

---

## 6.7 output_file：自动保存到文件

```python
output_file: str | None = Field(default=None)
create_directory: bool | None = Field(default=True)
```

### 6.7.1 路径安全验证

```python
@field_validator("output_file")
@classmethod
def output_file_validation(cls, value: str | None) -> str | None:
    if value is None:
        return value
    if ".." in value:
        raise ValueError("Path traversal attempts are not allowed in output_file paths")
    if value.startswith(("~", "$")):
        raise ValueError("Shell expansion characters are not allowed in output_file paths")
    if any(char in value for char in ["|", ">", "<", "&", ";"]):
        raise ValueError("Shell special characters are not allowed in output_file paths")
    # ...
```

防止路径遍历和命令注入——对可能接收 LLM 输出的系统至关重要。

### 6.7.2 保存逻辑

任务执行完成后自动保存：

```python
if self.output_file:
    content = (
        json_output if json_output
        else (pydantic_output.model_dump_json() if pydantic_output else result)
    )
    self._save_file(content)
```

`_save_file()` 的实现：

```python
def _save_file(self, result: dict[str, Any] | str | Any) -> None:
    resolved_path = Path(self.output_file).expanduser().resolve()
    directory = resolved_path.parent

    if self.create_directory and not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    with resolved_path.open("w", encoding="utf-8") as file:
        if isinstance(result, dict):
            json.dump(result, file, ensure_ascii=False, indent=2)
        else:
            file.write(str(result))
```

字典输出使用 `json.dump`（`ensure_ascii=False` 支持中文），统一 UTF-8 编码，`create_directory=True` 时自动创建多级目录。

### 6.7.3 路径模板插值

`output_file` 支持根据 kickoff inputs 动态生成路径：

```python
if self.output_file is not None:
    self.output_file = interpolate_only(
        input_string=self._original_output_file, inputs=inputs
    )
```

例如 `output_file="reports/{topic}.json"` 配合 `inputs={"topic": "AI"}` 会生成 `reports/AI.json`。

---

## 6.8 TaskOutput 的完整生命周期

```
Agent 执行 → 返回 result
    ↓
根据 result 类型和 Task 配置确定 raw/pydantic/json_dict
    ↓
创建 TaskOutput(raw, pydantic, json_dict, agent, output_format, messages)
    ↓
[可选] Guardrail 验证
    ↓
存储为 task.output → 触发 task.callback / crew.task_callback
    ↓
[可选] output_file → _save_file()
    ↓
发射 TaskCompletedEvent → 被 _execute_tasks() 收集
    ↓
最终由 _create_crew_output() 汇总为 CrewOutput
```

---

## 6.9 设计模式总结

**三级表示模式**：每个输出都有 raw/pydantic/json_dict 三种表示，raw 始终存在。保证向后兼容、渐进增强、容错降级。

**类型一致性**：`TaskOutput` 和 `CrewOutput` 共享相同核心字段和 API（`to_dict()`、`__str__()`、`json`），降低学习成本。

**关注点分离**：格式确定（`OutputFormat`）、数据承载（`TaskOutput`/`CrewOutput`）、转换（`_export_output()`）、持久化（`_save_file()`）各自独立。

---

## 本章要点

- **OutputFormat 定义三种格式**：`RAW`（字符串）、`JSON`（字典）、`PYDANTIC`（模型实例）
- **TaskOutput 始终包含 `raw` 字段**，可选包含 `pydantic` 和 `json_dict`，并记录 agent、output_format、messages 等元数据
- **CrewOutput 的核心输出取自最后一个有效任务**，同时包含全部 `tasks_output` 列表和 `token_usage` 统计
- **`output_pydantic` 和 `output_json` 都接受 Pydantic 模型类**，前者输出模型实例后者输出字典，两者互斥
- **输出转换两条路径**：LLM 直接返回 BaseModel，或返回字符串后通过 `convert_to_model()` 解析
- **`output_file` 具有安全验证**（防路径遍历、防 shell 注入）、自动建目录、模板插值
- **UsageMetrics 追踪五类指标**，通过累加模式汇总所有 Agent 使用量，`cached_prompt_tokens` 对成本优化有价值
- **CrewOutput 的 `__getitem__` 支持字典式访问**，统一 Pydantic 和 JSON 输出的使用方式
- **输出系统遵循"三级表示 + 优雅降级"**：raw 兜底，pydantic/json_dict 渐进增强
