# 第 20 章　Security 与 Telemetry

安全性和可观测性是生产级 Agent 框架不可或缺的两个维度。CrewAI 通过 `crewai/security/` 和 `crewai/telemetry/` 两个模块，为 Agent 系统提供了身份标识、安全配置和匿名遥测的完整基础设施。本章将深入分析这两个模块的设计与实现。

## 20.1 Security 模块概述

`crewai/security/` 目录结构简洁而精确：

```
security/
├── __init__.py
├── constants.py         # 安全常量（UUID namespace）
├── fingerprint.py       # Agent 指纹标识
└── security_config.py   # 安全配置
```

Security 模块的核心职责是为 Crew、Agent、Task 等实体提供唯一身份标识（Fingerprint），这些标识贯穿事件系统和遥测数据，实现端到端的可追踪性。

## 20.2 AgentFingerprint：实体身份标识

### 20.2.1 Fingerprint 类定义

`security/fingerprint.py` 中的 `Fingerprint` 类是整个安全体系的基石：

```python
class Fingerprint(BaseModel):
    _uuid_str: str = PrivateAttr(default_factory=lambda: str(uuid4()))
    _created_at: datetime = PrivateAttr(default_factory=datetime.now)
    metadata: Annotated[dict[str, Any], BeforeValidator(_validate_metadata)] = Field(
        default_factory=dict
    )

    @property
    def uuid_str(self) -> str:
        return self._uuid_str

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def uuid(self) -> UUID:
        return UUID(self.uuid_str)
```

设计要点：
- `_uuid_str` 和 `_created_at` 使用 `PrivateAttr`，防止被 Pydantic 序列化和外部修改
- 默认生成 UUID v4（随机），但也支持基于 seed 的确定性生成
- `metadata` 字段经过 `_validate_metadata` 验证，限制嵌套深度为 1 层且总大小不超过 10KB

### 20.2.2 确定性 UUID 生成

当需要跨运行保持一致的标识时，可以使用 seed 生成确定性 UUID：

```python
@classmethod
def _generate_uuid(cls, seed: str) -> str:
    if not seed.strip():
        raise ValueError("Seed cannot be empty or whitespace")
    return str(uuid5(CREW_AI_NAMESPACE, seed))
```

这里使用了 UUID v5（SHA-1 based），结合 CrewAI 专用的 namespace UUID：

```python
# security/constants.py
CREW_AI_NAMESPACE: UUID = UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
```

`CREW_AI_NAMESPACE` 是一个固定的 UUID v1 常量，作为 UUID v5 的 namespace。这意味着相同的 seed 在任何环境中都会生成相同的 Fingerprint UUID，非常适合以下场景：
- Agent 的 role 名作为 seed，确保相同角色的 Agent 有稳定的标识
- Task 的 key 作为 seed，用于跨运行的 Task 关联

### 20.2.3 工厂方法与序列化

```python
@classmethod
def generate(cls, seed: str | None = None,
             metadata: dict[str, Any] | None = None) -> Self:
    fingerprint = cls(metadata=metadata or {})
    if seed:
        fingerprint.__dict__["_uuid_str"] = cls._generate_uuid(seed)
    return fingerprint

def to_dict(self) -> dict[str, Any]:
    return {
        "uuid_str": self.uuid_str,
        "created_at": self.created_at.isoformat(),
        "metadata": self.metadata,
    }

@classmethod
def from_dict(cls, data: dict[str, Any]) -> Self:
    if not data:
        return cls()
    fingerprint = cls(metadata=data.get("metadata", {}))
    if "uuid_str" in data:
        fingerprint.__dict__["_uuid_str"] = data["uuid_str"]
    if "created_at" in data and isinstance(data["created_at"], str):
        fingerprint.__dict__["_created_at"] = datetime.fromisoformat(data["created_at"])
    return fingerprint
```

注意 `from_dict` 通过直接操作 `__dict__` 绕过 `PrivateAttr` 的限制来恢复序列化数据。`generate` 工厂方法同样使用此技巧。

### 20.2.4 Metadata 验证

```python
def _validate_metadata(v: Any) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise ValueError("Metadata must be a dictionary")

    for key, value in v.items():
        if not isinstance(key, str):
            raise ValueError(f"Metadata keys must be strings, got {type(key)}")
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if not isinstance(nested_key, str):
                    raise ValueError(...)
                if isinstance(nested_value, dict):
                    raise ValueError("Metadata can only be nested one level deep")

    # 防止 DoS 攻击
    if len(str(v)) > 10_000:
        raise ValueError("Metadata size exceeds maximum allowed (10KB)")

    return v
```

三层防御：
1. **类型约束**：所有 key 必须是 string
2. **深度限制**：最多一层嵌套，防止过度复杂的数据结构
3. **大小限制**：总大小不超过 10KB，防止内存滥用

### 20.2.5 相等性与哈希

```python
def __eq__(self, other: Any) -> bool:
    if type(other) is Fingerprint:
        return self.uuid_str == other.uuid_str
    return False

def __hash__(self) -> int:
    return hash(self.uuid_str)
```

使用 `type(other) is Fingerprint` 而非 `isinstance`，确保只有精确的 `Fingerprint` 类型才能比较，子类不参与比较。这是一个严格的安全决策。

## 20.3 SecurityConfig：安全配置

`security/security_config.py` 提供统一的安全配置入口：

```python
class SecurityConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fingerprint: Fingerprint = Field(
        default_factory=Fingerprint,
        description="Unique identifier for the component"
    )

    @field_validator("fingerprint", mode="before")
    @classmethod
    def validate_fingerprint(cls, v: Any) -> Fingerprint:
        if v is None:
            return Fingerprint()
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Fingerprint seed cannot be empty")
            return Fingerprint.generate(seed=v)
        if isinstance(v, dict):
            return Fingerprint.from_dict(v)
        if isinstance(v, Fingerprint):
            return v
        raise ValueError(f"Invalid fingerprint type: {type(v)}")
```

`validate_fingerprint` 的多态处理非常灵活：
- `None`：创建新的随机 Fingerprint
- `str`：作为 seed 生成确定性 Fingerprint
- `dict`：从序列化数据恢复
- `Fingerprint`：直接使用

这意味着用户可以用多种方式初始化安全配置：

```python
# 随机指纹
config = SecurityConfig()

# 基于角色名的确定性指纹
config = SecurityConfig(fingerprint="research_agent")

# 从持久化数据恢复
config = SecurityConfig(fingerprint={"uuid_str": "...", "metadata": {...}})
```

`SecurityConfig` 的注释中标记了未来计划（`TODO`）：
- Authentication credentials：Agent 级别的认证凭据
- Scoping rules：Agent 的权限范围控制
- Impersonation/delegation tokens：代理执行的 Token 机制

这表明当前的 `SecurityConfig` 是一个最小化但可扩展的设计。

## 20.4 Fingerprint 在系统中的传播

Fingerprint 不仅是一个静态标识，它在整个系统中流动：

### 在事件系统中

每个 `BaseEvent` 都携带 `source_fingerprint` 和 `source_type` 字段。事件的构造函数或 validator 自动从源实体提取 Fingerprint：

```python
# crew_events.py
class CrewBaseEvent(BaseEvent):
    def set_crew_fingerprint(self) -> None:
        if self.crew and hasattr(self.crew, "fingerprint"):
            self.source_fingerprint = self.crew.fingerprint.uuid_str
            self.source_type = "crew"
            if self.crew.fingerprint.metadata:
                self.fingerprint_metadata = self.crew.fingerprint.metadata

# agent_events.py
class AgentExecutionStartedEvent(BaseEvent):
    @model_validator(mode="after")
    def set_fingerprint_data(self):
        if hasattr(self.agent, "fingerprint"):
            self.source_fingerprint = self.agent.fingerprint.uuid_str
            self.source_type = "agent"
        return self
```

### 在 Telemetry 中

Telemetry span 中附加 Fingerprint 作为 attribute：

```python
# telemetry/utils.py
def add_agent_fingerprint_to_span(span, agent, add_attribute_fn):
    if agent and hasattr(agent, "fingerprint") and agent.fingerprint:
        add_attribute_fn(span, "agent_fingerprint", agent.fingerprint.uuid_str)
        if hasattr(agent, "role"):
            add_attribute_fn(span, "agent_role", agent.role)
```

### 在 A2A 事件中

A2A 事件将 Agent 和 Task 的 Fingerprint 作为默认的 source 信息：

```python
class A2AEventBase(BaseEvent):
    @model_validator(mode="before")
    @classmethod
    def extract_task_and_agent_metadata(cls, data):
        if agent := data.get("from_agent"):
            data.setdefault("source_fingerprint", str(agent.id))
            data.setdefault("source_type", "agent")
            data.setdefault("fingerprint_metadata", {
                "agent_id": str(agent.id),
                "agent_role": agent.role,
            })
        return data
```

## 20.5 Telemetry 模块概述

`crewai/telemetry/` 目录实现了基于 OpenTelemetry 的匿名遥测：

```
telemetry/
├── __init__.py
├── constants.py     # 遥测端点配置
├── telemetry.py     # Telemetry 主类
└── utils.py         # span 工具函数
```

## 20.6 Telemetry 类：OpenTelemetry 集成

### 20.6.1 初始化与安全导出

`Telemetry` 是 Singleton 模式，初始化时创建 OpenTelemetry TracerProvider：

```python
class Telemetry:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> Self:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.ready = False
        self.trace_set = False
        self._initialized = True

        if self._is_telemetry_disabled():
            return

        self.resource = Resource(
            attributes={SERVICE_NAME: CREWAI_TELEMETRY_SERVICE_NAME},
        )
        self.provider = TracerProvider(resource=self.resource)
        processor = BatchSpanProcessor(
            SafeOTLPSpanExporter(
                endpoint=f"{CREWAI_TELEMETRY_BASE_URL}/v1/traces",
                timeout=30,
            )
        )
        self.provider.add_span_processor(processor)
        self.ready = True
```

`SafeOTLPSpanExporter` 包装了标准的 OTLP exporter，防止导出失败影响应用：

```python
class SafeOTLPSpanExporter(OTLPSpanExporter):
    def export(self, spans: Any) -> SpanExportResult:
        try:
            return super().export(spans)
        except Exception as e:
            logger.error(e)
            return SpanExportResult.FAILURE
```

遥测端点配置在 `constants.py` 中：

```python
CREWAI_TELEMETRY_BASE_URL: Final[str] = "https://telemetry.crewai.com:4319"
CREWAI_TELEMETRY_SERVICE_NAME: Final[str] = "crewAI-telemetry"
```

### 20.6.2 遥测开关

用户可以通过三种环境变量禁用遥测：

```python
@classmethod
def _is_telemetry_disabled(cls) -> bool:
    return (
        os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true"
        or os.getenv("CREWAI_DISABLE_TELEMETRY", "false").lower() == "true"
        or os.getenv("CREWAI_DISABLE_TRACKING", "false").lower() == "true"
    )
```

- `OTEL_SDK_DISABLED`：OpenTelemetry 标准的全局开关
- `CREWAI_DISABLE_TELEMETRY`：CrewAI 专用的遥测开关
- `CREWAI_DISABLE_TRACKING`：向后兼容的遥测开关

每次遥测操作前都会检查开关状态：

```python
def _should_execute_telemetry(self) -> bool:
    return self.ready and not self._is_telemetry_disabled()

def _safe_telemetry_operation(self, operation):
    if not self._should_execute_telemetry():
        return None
    try:
        return operation()
    except Exception as e:
        logger.debug(f"Telemetry operation failed: {e}")
        return None
```

所有遥测操作都包装在 `_safe_telemetry_operation` 中，确保任何遥测故障都不会影响业务逻辑。

### 20.6.3 信号处理与优雅关闭

Telemetry 注册了系统信号处理器，在收到终止信号时发出事件并优雅关闭：

```python
def _register_shutdown_handlers(self) -> None:
    atexit.register(self._shutdown)

    self._register_signal_handler(signal.SIGTERM, SigTermEvent, shutdown=True)
    self._register_signal_handler(signal.SIGINT, SigIntEvent, shutdown=True)
    if hasattr(signal, "SIGHUP"):
        self._register_signal_handler(signal.SIGHUP, SigHupEvent, shutdown=False)
    if hasattr(signal, "SIGTSTP"):
        self._register_signal_handler(signal.SIGTSTP, SigTStpEvent, shutdown=False)
    if hasattr(signal, "SIGCONT"):
        self._register_signal_handler(signal.SIGCONT, SigContEvent, shutdown=False)
```

信号处理器会：
1. 发出对应的事件（如 `SigTermEvent`）
2. 若是终止信号，执行 `_shutdown` flush 所有 pending span
3. 链式调用原有的信号处理器

```python
def _shutdown(self) -> None:
    if not self.ready:
        return
    try:
        self.provider.force_flush(timeout_millis=5000)
        self.provider.shutdown()
        self.ready = False
    except Exception as e:
        logger.debug(f"Telemetry shutdown failed: {e}")
```

5 秒的 flush 超时确保进程不会因遥测问题而卡住。

## 20.7 Span 类型与数据脱敏

### 20.7.1 Crew Creation Span

`crew_creation` 记录 Crew 的创建，是最详细的 span 之一：

```python
def crew_creation(self, crew: Crew, inputs: dict[str, Any] | None) -> None:
    def _operation() -> None:
        tracer = trace.get_tracer("crewai.telemetry")
        span = tracer.start_span("Crew Created")
        self._add_attribute(span, "crewai_version", version("crewai"))
        self._add_attribute(span, "python_version", platform.python_version())
        add_crew_attributes(span, crew, self._add_attribute)
        self._add_attribute(span, "crew_number_of_tasks", len(crew.tasks))
        self._add_attribute(span, "crew_number_of_agents", len(crew.agents))
```

### 20.7.2 数据脱敏策略

CrewAI 在遥测数据中严格区分了"匿名数据"和"详细数据"。默认情况下，只收集结构性信息：

```python
# share_crew=False（默认）时只收集：
{
    "key": agent.key,
    "id": str(agent.id),
    "role": agent.role,
    "verbose?": agent.verbose,
    "max_iter": agent.max_iter,
    "llm": agent.llm.model,
    "delegation_enabled?": agent.allow_delegation,
    "tools_names": [sanitize_tool_name(tool.name) for tool in agent.tools or []],
}
```

只有用户显式设置 `share_crew=True` 时，才会收集以下敏感信息：

```python
# share_crew=True 时额外收集：
{
    "goal": agent.goal,
    "backstory": agent.backstory,
    "description": task.description,
    "expected_output": task.expected_output,
    "context": [task.description for task in task.context],
    "crew_inputs": json.dumps(inputs or {}),
}
```

Task 的输出也只在 `share_crew=True` 时记录：

```python
def task_ended(self, span, task, crew):
    def _operation():
        if crew.share_crew:
            self._add_attribute(span, "task_output", task.output.raw if task.output else "")
        close_span(span)
```

Tool 名称通过 `sanitize_tool_name` 进行清洗，避免用户自定义 tool 名中的敏感信息泄露。

### 20.7.3 完整的 Span 追踪链

从 Crew kickoff 到 LLM 调用的完整追踪链如下：

```
Crew Created (一次性)
    ├─ crewai_version, python_version
    ├─ crew_key, crew_id, crew_fingerprint
    ├─ crew_number_of_tasks, crew_number_of_agents
    └─ crew_agents[], crew_tasks[] (脱敏)

Crew Execution (share_crew=True 时)
    ├─ crew_agents[] (含 goal, backstory)
    ├─ crew_tasks[] (含 description, expected_output)
    └─ crew_inputs

Task Created
    ├─ crew_key, crew_id, task_key, task_id
    ├─ task_fingerprint, agent_fingerprint
    └─ formatted_description (share_crew=True)

Task Execution
    ├─ crew_key, crew_id, task_key, task_id
    ├─ task_fingerprint, agent_fingerprint
    └─ task_output (share_crew=True)

Tool Usage
    ├─ tool_name, attempts, llm
    └─ agent_fingerprint

Tool Repeated Usage
    ├─ tool_name, attempts, llm
    └─ (异常使用预警)

Flow Creation / Execution
    ├─ flow_name, node_names
    └─ (Flow 级别追踪)

Human Feedback
    ├─ event_type, has_routing, num_outcomes
    └─ feedback_provided, outcome
```

## 20.8 Telemetry 工具函数

`telemetry/utils.py` 提供了一组标准化的 span attribute 添加函数：

```python
def add_crew_attributes(span, crew, add_attribute_fn, include_fingerprint=True):
    add_attribute_fn(span, "crew_key", crew.key)
    add_attribute_fn(span, "crew_id", str(crew.id))
    if include_fingerprint and hasattr(crew, "fingerprint") and crew.fingerprint:
        add_attribute_fn(span, "crew_fingerprint", crew.fingerprint.uuid_str)

def add_task_attributes(span, task, add_attribute_fn, include_fingerprint=True):
    add_attribute_fn(span, "task_key", task.key)
    add_attribute_fn(span, "task_id", str(task.id))
    if include_fingerprint and hasattr(task, "fingerprint") and task.fingerprint:
        add_attribute_fn(span, "task_fingerprint", task.fingerprint.uuid_str)

def add_crew_and_task_attributes(span, crew, task, add_attribute_fn, ...):
    add_crew_attributes(span, crew, add_attribute_fn, include_fingerprints)
    add_task_attributes(span, task, add_attribute_fn, include_fingerprints)

def close_span(span: Span) -> None:
    span.set_status(Status(StatusCode.OK))
    span.end()
```

所有函数都接受 `add_attribute_fn` 参数而非直接操作 span，这使得 attribute 添加也经过 `_safe_telemetry_operation` 包装，进一步增强容错性。

`include_fingerprint` 参数允许调用方选择性地省略 Fingerprint，在某些高频 span（如 test result）中减少数据量。

## 20.9 Events 与 Telemetry 的集成

EventListener 是连接 Events 系统和 Telemetry 的桥梁：

```python
class EventListener(BaseEventListener):
    def __init__(self):
        super().__init__()
        self._telemetry = Telemetry()
        self._telemetry.set_tracer()

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event):
            # 1. 控制台输出
            self.formatter.handle_crew_started(event.crew_name, source.id)
            # 2. 创建 Telemetry span
            source._execution_span = self._telemetry.crew_execution_span(
                source, event.inputs
            )

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event):
            # 关闭 Telemetry span
            self._telemetry.end_crew(source, event.output.raw)
            # 控制台输出
            self.formatter.handle_crew_status(...)
```

这种设计将事件的产生（emit）和消费（handler）完全解耦。Telemetry 只是事件的众多消费者之一，用户可以自由添加其他消费者（如外部监控系统、日志聚合等）而不影响现有行为。

## 20.10 安全与遥测的设计原则

从整体架构来看，CrewAI 的安全与遥测设计遵循以下原则：

**最小侵入原则**：Fingerprint 通过 `PrivateAttr` 隐藏实现细节，SecurityConfig 自动生成默认配置，用户无需任何额外代码即可获得基础的安全标识。

**故障隔离原则**：所有遥测操作都包装在 `_safe_telemetry_operation` 中，遥测系统的任何故障不会影响 Agent 的正常执行。

**数据最小化原则**：默认只收集结构性信息（agent count、task count、tool names），不收集 prompt、backstory、task description 等潜在敏感数据。

**可选增强原则**：`share_crew=True` 是一个显式的 opt-in 机制，用户主动选择共享更多数据以获得更好的分析支持。

**确定性可追踪原则**：通过 UUID v5 + CREW_AI_NAMESPACE 的组合，相同 Agent/Task 在不同运行中可以被关联追踪，同时 Fingerprint 在事件和 span 中的传播确保了端到端的可追溯性。

## 本章要点

- `Fingerprint` 是 CrewAI 实体的唯一身份标识，支持随机生成（UUID v4）和基于 seed 的确定性生成（UUID v5 + CREW_AI_NAMESPACE），metadata 有深度限制（1 层）和大小限制（10KB）
- `SecurityConfig` 通过 `field_validator` 实现多态 Fingerprint 初始化：接受 None、str、dict 或 Fingerprint 对象
- Fingerprint 在三个维度传播：事件系统（`source_fingerprint`）、Telemetry span（`agent_fingerprint`/`task_fingerprint`）、A2A 事件（`fingerprint_metadata`）
- `Telemetry` 类基于 OpenTelemetry TracerProvider + OTLP Exporter 实现，使用 `SafeOTLPSpanExporter` 确保导出故障不影响应用
- 遥测开关支持 `OTEL_SDK_DISABLED`、`CREWAI_DISABLE_TELEMETRY`、`CREWAI_DISABLE_TRACKING` 三种环境变量
- 数据脱敏策略严格区分匿名数据（默认）和详细数据（`share_crew=True`），匿名模式下不收集 prompt、goal、backstory、task output 等敏感信息
- 信号处理覆盖 SIGTERM/SIGINT（触发 shutdown）和 SIGHUP/SIGTSTP/SIGCONT（仅发出事件），确保遥测数据在进程退出前 flush 完成（5 秒超时）
- 所有遥测操作通过 `_safe_telemetry_operation` 包装，配合 `_should_execute_telemetry` 双重检查，实现零影响的故障隔离
- Telemetry span 覆盖 Crew 生命周期全链路：Crew Created -> Task Created -> Task Execution -> Tool Usage -> Crew Execution，每层都携带 Fingerprint 用于关联分析
