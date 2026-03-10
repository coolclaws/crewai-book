# 第 19 章　Events 系统：事件总线与可观测性

一个完整的 Agent 框架需要让使用者清楚地知道"正在发生什么"。CrewAI 通过 `crewai/events/` 模块实现了一套完整的事件系统，覆盖从 Crew 启动到 LLM 流式输出的每一个关键环节。本章将深入分析事件总线的架构、事件类型体系、监听器机制以及 Rich 终端格式化输出。

## 19.1 架构总览

Events 系统的核心组件如下：

```
                    ┌──────────────────────┐
                    │  CrewAIEventsBus     │  (Singleton)
                    │  ┌────────────────┐  │
   emit(source,     │  │ sync_handlers  │  │  ThreadPoolExecutor
   event) ─────────►│  │ async_handlers │  │  asyncio event loop
                    │  │ dependencies   │  │  dependency graph
                    │  └────────────────┘  │
                    └──────────┬───────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    EventListener       BaseEventListener    Custom Listeners
    (built-in)          (abstract base)
    ┌─────────────┐
    │ Telemetry   │
    │ Console     │
    │ Formatter   │
    └─────────────┘
```

## 19.2 BaseEvent：事件基类

所有事件继承自 `BaseEvent`（定义在 `events/base_events.py`），它基于 Pydantic BaseModel 构建：

```python
class BaseEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    type: str
    source_fingerprint: str | None = None
    source_type: str | None = None
    fingerprint_metadata: dict[str, Any] | None = None

    task_id: str | None = None
    task_name: str | None = None
    agent_id: str | None = None
    agent_role: str | None = None

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_event_id: str | None = None
    previous_event_id: str | None = None
    triggered_by_event_id: str | None = None
    started_event_id: str | None = None
    emission_sequence: int | None = None
```

每个事件携带丰富的上下文信息：

- **`event_id`**：UUID v4，事件唯一标识
- **`parent_event_id`**：父事件 ID，形成嵌套的事件作用域树
- **`previous_event_id`**：上一个发出的事件 ID，形成线性事件链
- **`triggered_by_event_id`**：触发当前执行的事件 ID，用于因果链追踪
- **`started_event_id`**：在结束事件上指向对应的开始事件 ID
- **`emission_sequence`**：全局递增的发射序号
- **`source_fingerprint`** 和 **`source_type`**：发射源的 Fingerprint 标识

这套 ID 体系使得事件流可以被还原为完整的执行树（tree）、线性链（chain）和因果图（causal graph）。

`emission_sequence` 的计数器使用 `contextvars.ContextVar` 实现上下文隔离：

```python
_emission_counter: contextvars.ContextVar[Iterator[int]] = contextvars.ContextVar(
    "_emission_counter"
)

def get_next_emission_sequence() -> int:
    return next(_get_or_create_counter())
```

## 19.3 CrewAIEventsBus：全局事件总线

`events/event_bus.py` 中的 `CrewAIEventsBus` 是整个事件系统的核心，采用 Singleton 模式：

```python
class CrewAIEventsBus:
    _instance: Self | None = None
    _instance_lock: threading.RLock = threading.RLock()

    def __new__(cls) -> Self:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
```

### 19.3.1 内部架构

事件总线内部维护以下关键状态：

```python
def _initialize(self) -> None:
    self._rwlock = RWLock()
    self._sync_handlers: dict[type[BaseEvent], SyncHandlerSet] = {}
    self._async_handlers: dict[type[BaseEvent], AsyncHandlerSet] = {}
    self._handler_dependencies: dict[type[BaseEvent], dict[Handler, list[Depends]]] = {}
    self._execution_plan_cache: dict[type[BaseEvent], ExecutionPlan] = {}
    self._sync_executor = ThreadPoolExecutor(
        max_workers=10, thread_name_prefix="CrewAISyncHandler",
    )
    self._loop = asyncio.new_event_loop()
    self._loop_thread = threading.Thread(
        target=self._run_loop, name="CrewAIEventsLoop", daemon=True,
    )
    self._loop_thread.start()
```

关键设计决策：
- **RWLock**：使用读写锁分离 handler 的注册（写）和查询（读），允许多个事件并发发射
- **双 Handler 集合**：同步和异步 handler 分开存储，使用 `frozenset` 实现不可变集合
- **专用 asyncio event loop**：在 daemon 线程中运行独立的事件循环，避免与主线程的事件循环冲突
- **ThreadPoolExecutor**：10 个工作线程处理同步 handler，确保不阻塞主线程

### 19.3.2 Handler 注册

使用装饰器 API 注册事件处理器：

```python
def on(self, event_type: type[BaseEvent],
       depends_on: Depends | list[Depends] | None = None):
    def decorator(handler):
        deps = None
        if depends_on is not None:
            deps = [depends_on] if isinstance(depends_on, Depends) else depends_on
        self._register_handler(event_type, handler, dependencies=deps)
        return handler
    return decorator
```

使用示例：

```python
@crewai_event_bus.on(LLMCallStartedEvent)
def setup_context(source, event):
    print("Setting up context")

@crewai_event_bus.on(LLMCallStartedEvent, depends_on=Depends(setup_context))
def process(source, event):
    print("Processing (runs after setup_context)")
```

`Depends` 机制允许声明 handler 之间的依赖关系。当存在依赖时，事件总线会构建执行计划（execution plan），按拓扑排序分层执行。

### 19.3.3 事件发射

`emit` 方法是事件发射的入口，它处理事件上下文管理和 handler 调度：

```python
def emit(self, source: Any, event: BaseEvent) -> Future[None] | None:
    # 1. 设置事件链信息
    event.previous_event_id = get_last_event_id()
    event.triggered_by_event_id = get_triggering_event_id()
    event.emission_sequence = get_next_emission_sequence()

    # 2. 管理事件作用域
    if event.parent_event_id is None:
        event_type_name = event.type
        if event_type_name in SCOPE_ENDING_EVENTS:
            event.parent_event_id = get_enclosing_parent_id()
            popped = pop_event_scope()
            # 验证事件配对...
        elif event_type_name in SCOPE_STARTING_EVENTS:
            event.parent_event_id = get_current_parent_id()
            push_event_scope(event.event_id, event_type_name)
        else:
            event.parent_event_id = get_current_parent_id()

    set_last_event_id(event.event_id)

    # 3. 根据是否有依赖选择执行策略
    if has_dependencies:
        return asyncio.run_coroutine_threadsafe(
            self._emit_with_dependencies(source, event), self._loop
        )

    # 4. LLMStreamChunkEvent 特殊处理：同步执行保序
    if sync_handlers:
        if event_type is LLMStreamChunkEvent:
            self._call_handlers(source, event, sync_handlers)
        else:
            ctx = contextvars.copy_context()
            sync_future = self._sync_executor.submit(
                ctx.run, self._call_handlers, source, event, sync_handlers
            )

    if async_handlers:
        return asyncio.run_coroutine_threadsafe(
            self._acall_handlers(source, event, async_handlers), self._loop
        )
```

注意几个精妙之处：
- `LLMStreamChunkEvent` 被特殊对待，同步执行以保证 chunk 的顺序
- `contextvars.copy_context()` 确保线程池中的 handler 能访问到正确的上下文变量
- 返回 `Future` 允许调用方选择性等待 handler 执行完成

### 19.3.4 事件作用域管理

`event_context.py` 定义了事件作用域的嵌套机制。系统维护了配对的起始/结束事件集合：

```python
SCOPE_STARTING_EVENTS: frozenset[str] = frozenset({
    "flow_started", "method_execution_started",
    "crew_kickoff_started", "agent_execution_started",
    "task_started", "llm_call_started",
    "tool_usage_started", "a2a_delegation_started",
    # ... 共 27 种
})

SCOPE_ENDING_EVENTS: frozenset[str] = frozenset({
    "flow_finished", "method_execution_finished",
    "crew_kickoff_completed", "agent_execution_completed",
    "task_completed", "llm_call_completed",
    "tool_usage_finished", "a2a_delegation_completed",
    # ... 共 35 种
})

VALID_EVENT_PAIRS: dict[str, str] = {
    "crew_kickoff_completed": "crew_kickoff_started",
    "task_completed": "task_started",
    "llm_call_completed": "llm_call_started",
    # ...
}
```

作用域使用 `contextvars.ContextVar` 存储的不可变 tuple 栈实现，并配有深度限制和配对校验：

```python
def push_event_scope(event_id: str, event_type: str = "") -> None:
    config = _event_context_config.get() or _default_config
    stack = _event_id_stack.get()
    if 0 < config.max_stack_depth <= len(stack):
        raise StackDepthExceededError(...)
    _event_id_stack.set((*stack, (event_id, event_type)))
```

这确保了一次典型执行中事件的 parent-child 关系如下：

```
crew_kickoff_started
  └─ task_started
       └─ agent_execution_started
            └─ llm_call_started
                 └─ tool_usage_started
                      └─ tool_usage_finished
                 └─ llm_call_completed
            └─ agent_execution_completed
       └─ task_completed
  └─ crew_kickoff_completed
```

### 19.3.5 依赖感知的执行计划

当 handler 声明了依赖关系时，`_emit_with_dependencies` 使用缓存的执行计划：

```python
async def _emit_with_dependencies(self, source, event):
    event_type = type(event)

    # 尝试从缓存获取执行计划
    cached_plan = self._execution_plan_cache.get(event_type)
    if cached_plan is None:
        # 构建执行计划（拓扑排序）
        cached_plan = build_execution_plan(all_handlers, dependencies)
        self._execution_plan_cache[event_type] = cached_plan

    # 按层级执行
    for level in cached_plan:
        level_sync = frozenset(h for h in level if h in sync_handlers)
        level_async = frozenset(h for h in level if h in async_handlers)

        if level_sync:
            # 同步 handler 在线程池中执行
            future = self._sync_executor.submit(
                ctx.run, self._call_handlers, source, event, level_sync
            )
            await asyncio.get_running_loop().run_in_executor(None, future.result)

        if level_async:
            await self._acall_handlers(source, event, level_async)
```

执行计划是一个二维列表，每一层（level）内的 handler 可以并行执行，层与层之间串行执行。

### 19.3.6 flush 与 shutdown

```python
def flush(self, timeout: float | None = 30.0) -> bool:
    with self._futures_lock:
        futures_to_wait = list(self._pending_futures)
    if not futures_to_wait:
        return True
    done, not_done = wait_futures(futures_to_wait, timeout=timeout)
    return len(not_done) == 0
```

`flush` 确保所有 pending 的 handler 执行完成，默认 30 秒超时。在 Crew `kickoff` 结束时调用，保证所有事件处理完毕再返回结果。

`shutdown` 在进程退出时通过 `atexit.register` 自动调用：

```python
crewai_event_bus: Final[CrewAIEventsBus] = CrewAIEventsBus()
atexit.register(crewai_event_bus.shutdown)
```

## 19.4 事件类型体系

`events/types/` 目录包含了 16 个事件类型模块，覆盖系统的所有层面。

### 19.4.1 Crew 事件

```python
class CrewKickoffStartedEvent(CrewBaseEvent):
    inputs: dict[str, Any] | None
    type: str = "crew_kickoff_started"

class CrewKickoffCompletedEvent(CrewBaseEvent):
    output: Any
    type: str = "crew_kickoff_completed"
    total_tokens: int = 0

class CrewKickoffFailedEvent(CrewBaseEvent):
    error: str
    type: str = "crew_kickoff_failed"
```

`CrewBaseEvent` 自动从 Crew 实例提取 Fingerprint 信息：

```python
class CrewBaseEvent(BaseEvent):
    crew_name: str | None
    crew: Crew | None = None

    def set_crew_fingerprint(self) -> None:
        if self.crew and hasattr(self.crew, "fingerprint"):
            self.source_fingerprint = self.crew.fingerprint.uuid_str
            self.source_type = "crew"
```

### 19.4.2 Task 事件

```python
class TaskStartedEvent(BaseEvent):
    type: str = "task_started"
    context: str | None
    task: Any | None = None

class TaskCompletedEvent(BaseEvent):
    output: TaskOutput
    type: str = "task_completed"

class TaskFailedEvent(BaseEvent):
    error: str
    type: str = "task_failed"
```

### 19.4.3 Agent 事件

```python
class AgentExecutionStartedEvent(BaseEvent):
    agent: BaseAgent
    task: Any
    tools: Sequence[BaseTool | CrewStructuredTool] | None
    task_prompt: str
    type: str = "agent_execution_started"

class AgentExecutionCompletedEvent(BaseEvent):
    agent: BaseAgent
    task: Any
    output: str
    type: str = "agent_execution_completed"
```

Agent 事件通过 `@model_validator` 自动提取 Fingerprint：

```python
@model_validator(mode="after")
def set_fingerprint_data(self):
    if hasattr(self.agent, "fingerprint") and self.agent.fingerprint:
        self.source_fingerprint = self.agent.fingerprint.uuid_str
        self.source_type = "agent"
    return self
```

### 19.4.4 LLM 事件

LLM 事件是最细粒度的事件，支持流式 chunk：

```python
class LLMCallStartedEvent(LLMEventBase):
    type: str = "llm_call_started"
    messages: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None

class LLMCallCompletedEvent(LLMEventBase):
    type: str = "llm_call_completed"
    response: Any
    call_type: LLMCallType  # TOOL_CALL or LLM_CALL

class LLMStreamChunkEvent(LLMEventBase):
    type: str = "llm_stream_chunk"
    chunk: str
    tool_call: ToolCall | None = None
    call_type: LLMCallType | None = None

class LLMThinkingChunkEvent(LLMEventBase):
    type: str = "llm_thinking_chunk"
    chunk: str
```

`LLMThinkingChunkEvent` 是专门为支持"思考模型"（thinking model）设计的事件。

### 19.4.5 Tool 事件

```python
class ToolUsageStartedEvent(ToolUsageEvent):
    type: str = "tool_usage_started"

class ToolUsageFinishedEvent(ToolUsageEvent):
    started_at: datetime
    finished_at: datetime
    from_cache: bool = False
    output: Any
    type: str = "tool_usage_finished"

class ToolUsageErrorEvent(ToolUsageEvent):
    error: Any
    type: str = "tool_usage_error"
```

`ToolUsageFinishedEvent` 携带 `from_cache` 字段，标识结果是否来自缓存。

### 19.4.6 Flow 事件

Flow 事件覆盖了从流程创建到人工反馈的完整生命周期：

```python
class FlowStartedEvent(FlowEvent):           # 流程开始
class FlowFinishedEvent(FlowEvent):           # 流程结束
class FlowPausedEvent(FlowEvent):             # 流程暂停（等待人工反馈）
class MethodExecutionStartedEvent(FlowEvent):  # 方法执行开始
class MethodExecutionPausedEvent(FlowEvent):   # 方法暂停
class HumanFeedbackRequestedEvent(FlowEvent):  # 请求人工反馈
class HumanFeedbackReceivedEvent(FlowEvent):   # 收到人工反馈
class FlowInputRequestedEvent(FlowEvent):      # Flow.ask() 请求输入
class FlowInputReceivedEvent(FlowEvent):       # 收到用户输入
```

### 19.4.7 A2A 事件

A2A 事件是最丰富的事件类别，定义在 `types/a2a_events.py` 中，包含约 20 种事件类型：

```python
class A2ADelegationStartedEvent(A2AEventBase)      # 委托开始
class A2ADelegationCompletedEvent(A2AEventBase)     # 委托完成
class A2AConversationStartedEvent(A2AEventBase)     # 多轮对话开始
class A2AMessageSentEvent(A2AEventBase)             # 消息发送
class A2AResponseReceivedEvent(A2AEventBase)        # 响应接收
class A2AConversationCompletedEvent(A2AEventBase)   # 对话结束
class A2APollingStartedEvent(A2AEventBase)          # Polling 开始
class A2AStreamingStartedEvent(A2AEventBase)        # Streaming 开始
class A2AStreamingChunkEvent(A2AEventBase)          # Streaming chunk
class A2AAgentCardFetchedEvent(A2AEventBase)        # AgentCard 获取
class A2AAuthenticationFailedEvent(A2AEventBase)    # 认证失败
class A2ATransportNegotiatedEvent(A2AEventBase)     # 传输协商
class A2AContextCreatedEvent(A2AEventBase)          # Context 创建
class A2AParallelDelegationStartedEvent(A2AEventBase) # 并行委托
# ... 等
```

### 19.4.8 其他事件类型

系统还定义了以下事件模块：
- **knowledge_events**：知识检索与查询事件
- **memory_events**：记忆存储与检索事件
- **mcp_events**：MCP 连接与 tool 执行事件
- **reasoning_events**：Agent 推理过程事件
- **llm_guardrail_events**：LLM 输出护栏事件
- **logging_events**：Agent 日志事件
- **system_events**：信号处理事件（SIGTERM、SIGINT 等）

## 19.5 BaseEventListener：监听器基类

```python
class BaseEventListener(ABC):
    verbose: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.setup_listeners(crewai_event_bus)
        crewai_event_bus.validate_dependencies()

    @abstractmethod
    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        ...
```

自定义监听器只需继承 `BaseEventListener` 并实现 `setup_listeners` 方法。构造函数自动完成两件事：
1. 调用 `setup_listeners` 注册所有 handler
2. 调用 `validate_dependencies` 验证依赖关系无环

## 19.6 内置 EventListener：Telemetry 与控制台输出

`events/event_listener.py` 中的 `EventListener` 是系统唯一的内置监听器，同时承担 Telemetry 采集和控制台输出两个职责：

```python
class EventListener(BaseEventListener):
    _telemetry: Telemetry = PrivateAttr(default_factory=lambda: Telemetry())
    logger: Logger = Logger(verbose=True, default_color=EMITTER_COLOR)

    def __init__(self) -> None:
        if not self._initialized:
            super().__init__()
            self._telemetry = Telemetry()
            self._telemetry.set_tracer()
            self.formatter = ConsoleFormatter(verbose=True)
            trace_listener = TraceCollectionListener()
            trace_listener.formatter = self.formatter
```

`setup_listeners` 方法注册了 40+ 个 handler，覆盖所有事件类型。以 Crew 事件为例：

```python
def setup_listeners(self, crewai_event_bus):
    @crewai_event_bus.on(CrewKickoffStartedEvent)
    def on_crew_started(source, event):
        self.formatter.handle_crew_started(event.crew_name or "Crew", source.id)
        source._execution_span = self._telemetry.crew_execution_span(
            source, event.inputs
        )

    @crewai_event_bus.on(TaskStartedEvent)
    def on_task_started(source, event):
        span = self._telemetry.task_started(crew=source.agent.crew, task=source)
        self.execution_spans[source] = span
        self.formatter.handle_task_started(source.id, task_name)
```

每个 handler 同时做两件事：
1. 调用 `self._telemetry` 记录 OpenTelemetry span
2. 调用 `self.formatter` 输出 Rich 格式化的终端信息

LLM 流式输出的处理展示了事件系统的实时性：

```python
@crewai_event_bus.on(LLMStreamChunkEvent)
def on_llm_stream_chunk(_, event):
    self.text_stream.write(event.chunk)
    self.text_stream.seek(self.next_chunk)
    self.text_stream.read()
    self.next_chunk = self.text_stream.tell()

    accumulated_text = self.text_stream.getvalue()
    self.formatter.handle_llm_stream_chunk(accumulated_text, event.call_type)
```

使用 `StringIO` 累积 chunk 文本，每收到新 chunk 就更新终端显示。

## 19.7 ConsoleFormatter：Rich 终端输出

`events/utils/console_formatter.py` 使用 Rich 库实现精美的终端输出，约 1700 行代码：

```python
class ConsoleFormatter:
    tool_usage_counts: ClassVar[dict[str, int]] = {}
    current_a2a_turn_count: int = 0

    def __init__(self, verbose: bool = False):
        self.console = Console(width=None)
        self.verbose = verbose
```

`ConsoleFormatter` 为每类事件提供专门的格式化方法（`handle_crew_started`、`handle_task_status`、`handle_llm_stream_chunk` 等），使用 Rich 的 `Panel`、`Text`、`Live` 等组件实现实时更新的终端 UI。

`ConsoleFormatter` 还维护了一些跨事件的状态，如 `tool_usage_counts`（统计 tool 调用次数）和 `current_a2a_turn_count`（追踪 A2A 对话轮次），用于在终端输出中提供更丰富的上下文信息。

## 19.8 scoped_handlers：测试支持

事件总线提供了 `scoped_handlers` 上下文管理器，用于测试中临时注册 handler：

```python
with crewai_event_bus.scoped_handlers():
    @crewai_event_bus.on(CrewKickoffStartedEvent)
    def temp_handler(source, event):
        captured_events.append(event)

    crew.kickoff()
    # temp_handler 只在此 scope 内有效
# scope 结束后，handler 自动移除，原有 handler 恢复
```

实现方式是在进入 scope 时保存当前所有 handler 的快照，退出时移除新增的 handler 并恢复原有 handler。

## 本章要点

- `CrewAIEventsBus` 是 Singleton 全局事件总线，维护独立的 asyncio event loop 和 `ThreadPoolExecutor`，支持同步/异步 handler 并行执行
- `BaseEvent` 携带完整的事件链信息（event_id、parent_event_id、previous_event_id、triggered_by_event_id），支持重建执行树、线性链和因果图
- 事件作用域通过 `SCOPE_STARTING_EVENTS`/`SCOPE_ENDING_EVENTS` 自动管理，使用 `contextvars` 栈实现嵌套，具有深度限制和配对校验
- `LLMStreamChunkEvent` 特殊处理为同步执行以保证 chunk 顺序，其他事件默认在线程池/异步循环中非阻塞执行
- Handler 支持 `Depends` 依赖声明，事件总线自动构建拓扑排序的执行计划并缓存
- 事件类型体系覆盖 Crew/Task/Agent/LLM/Tool/Flow/A2A/Knowledge/Memory/MCP/Reasoning/Guardrail/System 共 16 个模块，约 80+ 种事件类型
- 内置 `EventListener` 同时处理 Telemetry span 记录和 Rich 终端格式化输出
- `BaseEventListener` 抽象类提供自定义监听器的扩展点，构造时自动注册 handler 并验证依赖
- `scoped_handlers` 上下文管理器支持测试中的临时 handler 注册与自动清理
