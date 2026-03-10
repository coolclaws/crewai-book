# 第 5 章　Crew：团队编排与 Process 模式

`Crew` 是 CrewAI 框架的核心编排器，负责将 Agent 和 Task 组织成协作团队，根据 Process 模式决定任务执行方式，管理上下文传递和回调系统，并最终汇总 `CrewOutput`。本章以源码为主线深入分析其设计与实现。

---

## 5.1 Crew 类的整体结构

`Crew` 定义在 `crew.py` 中（约 2040 行），继承自 `FlowTrackable` 和 Pydantic `BaseModel`：

```python
class Crew(FlowTrackable, BaseModel):
    __hash__ = object.__hash__
    _execution_span: Any = PrivateAttr()
    _rpm_controller: RPMController = PrivateAttr()
    _logger: Logger = PrivateAttr()
    _cache_handler: InstanceOf[CacheHandler] = PrivateAttr(default_factory=CacheHandler)
    _memory: Any = PrivateAttr(default=None)
    _task_output_handler: TaskOutputStorageHandler = PrivateAttr(
        default_factory=TaskOutputStorageHandler
    )

    name: str | None = Field(default="crew")
    tasks: list[Task] = Field(default_factory=list)
    agents: list[BaseAgent] = Field(default_factory=list)
    process: Process = Field(default=Process.sequential)
    verbose: bool = Field(default=False)
    memory: bool | Any = Field(default=False)
    cache: bool = Field(default=True)
    manager_llm: str | InstanceOf[BaseLLM] | Any | None = Field(default=None)
    manager_agent: BaseAgent | None = Field(default=None)
    step_callback: Any | None = Field(default=None)
    task_callback: Any | None = Field(default=None)
    before_kickoff_callbacks: list[Callable] = Field(default_factory=list)
    after_kickoff_callbacks: list[Callable[[CrewOutput], CrewOutput]] = Field(default_factory=list)
    stream: bool = Field(default=False)
    planning: bool | None = Field(default=False)
    knowledge: Knowledge | None = Field(default=None)
```

`PrivateAttr` 存储运行时内部状态，`Field` 暴露用户可配置属性，接口简洁而内部状态充分隔离。

---

## 5.2 Process 模式：sequential 与 hierarchical

`Process` 定义在 `process.py` 中：

```python
class Process(str, Enum):
    sequential = "sequential"
    hierarchical = "hierarchical"
    # TODO: consensual = 'consensual'
```

### 5.2.1 Sequential 模式

默认模式。Task 按定义顺序逐一执行，每个 Task 必须预先绑定 Agent：

```python
@model_validator(mode="after")
def validate_tasks(self) -> Self:
    if self.process == Process.sequential:
        for task in self.tasks:
            if task.agent is None:
                raise PydanticCustomError(
                    "missing_agent_in_task",
                    "Sequential process error: Agent is missing in the task "
                    "with the following description: {description}",
                    {"description": task.description},
                )
    return self
```

执行入口极其简洁：

```python
def _run_sequential_process(self) -> CrewOutput:
    return self._execute_tasks(self.tasks)
```

### 5.2.2 Hierarchical 模式

引入 Manager Agent 动态分配任务给 Worker Agent。必须提供 `manager_llm` 或 `manager_agent`：

```python
@model_validator(mode="after")
def check_manager_llm(self) -> Self:
    if self.process == Process.hierarchical:
        if not self.manager_llm and not self.manager_agent:
            raise PydanticCustomError(
                "missing_manager_llm_or_manager_agent",
                "Attribute `manager_llm` or `manager_agent` is required "
                "when using hierarchical process.", {},
            )
        if self.manager_agent and self.agents.count(self.manager_agent) > 0:
            raise PydanticCustomError(
                "manager_agent_in_agents",
                "Manager agent should not be included in agents list.", {},
            )
    return self
```

Manager Agent 的创建揭示了关键设计决策：

```python
def _create_manager_agent(self) -> None:
    if self.manager_agent is not None:
        self.manager_agent.allow_delegation = True
        manager = self.manager_agent
        if manager.tools is not None and len(manager.tools) > 0:
            manager.tools = []
            raise Exception("Manager agent should not have tools")
    else:
        self.manager_llm = create_llm(self.manager_llm)
        i18n = get_i18n(prompt_file=self.prompt_file)
        manager = Agent(
            role=i18n.retrieve("hierarchical_manager_agent", "role"),
            goal=i18n.retrieve("hierarchical_manager_agent", "goal"),
            backstory=i18n.retrieve("hierarchical_manager_agent", "backstory"),
            tools=AgentTools(agents=self.agents).tools(),
            allow_delegation=True,
            llm=self.manager_llm,
        )
        self.manager_agent = manager
    manager.crew = self
```

Manager 不允许持有自己的工具（只能委派），自动创建时通过 `AgentTools(agents=self.agents).tools()` 获得委派能力，必须启用 `allow_delegation=True`。

在 hierarchical 模式下，所有任务都通过 Manager 执行：

```python
def _get_agent_to_use(self, task: Task) -> BaseAgent | None:
    if self.process == Process.hierarchical:
        return self.manager_agent
    return task.agent
```

---

## 5.3 kickoff()：启动执行

```python
def kickoff(
    self,
    inputs: dict[str, Any] | None = None,
    input_files: dict[str, FileInput] | None = None,
) -> CrewOutput | CrewStreamingOutput:
```

Streaming 模式通过包装普通执行流程实现，避免了全链路的 streaming 逻辑侵入。非 streaming 的核心流程：

```python
try:
    inputs = prepare_kickoff(self, inputs, input_files)

    if self.process == Process.sequential:
        result = self._run_sequential_process()
    elif self.process == Process.hierarchical:
        result = self._run_hierarchical_process()

    for after_callback in self.after_kickoff_callbacks:
        result = after_callback(result)

    self.usage_metrics = self.calculate_usage_metrics()
    return result
finally:
    if self._memory is not None and hasattr(self._memory, "drain_writes"):
        self._memory.drain_writes()
    clear_files(self.id)
```

`prepare_kickoff()` 在 `crews/utils.py` 中，按序完成：执行 `before_kickoff_callbacks`、发射事件、重置 handler、插值 inputs、设置 callbacks、初始化 agents、执行 planning。

---

## 5.4 _execute_tasks()：任务执行引擎

无论 sequential 还是 hierarchical 模式都调用此方法：

```python
def _execute_tasks(self, tasks: list[Task], start_index: int | None = 0,
                   was_replayed: bool = False) -> CrewOutput:
    task_outputs: list[TaskOutput] = []
    futures: list[tuple[Task, Future[TaskOutput], int]] = []
    last_sync_output: TaskOutput | None = None

    for task_index, task in enumerate(tasks):
        exec_data, task_outputs, last_sync_output = prepare_task_execution(
            self, task, task_index, start_index, task_outputs, last_sync_output)
        if exec_data.should_skip:
            continue

        if isinstance(task, ConditionalTask):
            skipped = self._handle_conditional_task(task, task_outputs, futures, task_index, was_replayed)
            if skipped:
                task_outputs.append(skipped)
                continue

        if task.async_execution:
            context = self._get_context(task, [last_sync_output] if last_sync_output else [])
            future = task.execute_async(agent=exec_data.agent, context=context, tools=exec_data.tools)
            futures.append((task, future, task_index))
        else:
            if futures:
                task_outputs = self._process_async_tasks(futures, was_replayed)
                futures.clear()
            context = self._get_context(task, task_outputs)
            task_output = task.execute_sync(agent=exec_data.agent, context=context, tools=exec_data.tools)
            task_outputs.append(task_output)

    if futures:
        task_outputs = self._process_async_tasks(futures, was_replayed)
    return self._create_crew_output(task_outputs)
```

关键设计：异步任务通过 `Future` 收集，遇到同步任务时先等待所有挂起的异步任务完成，确保数据一致性。`ConditionalTask` 根据前一个任务的输出决定是否执行。

---

## 5.5 Task 上下文传递

Task 的 `context` 字段默认值是 `NOT_SPECIFIED`（哨兵值），而不是 `None`：

```python
context: list[Task] | None | _NotSpecified = Field(
    default=NOT_SPECIFIED,
)
```

`_get_context()` 方法处理三种情况：

```python
@staticmethod
def _get_context(task: Task, task_outputs: list[TaskOutput]) -> str:
    if not task.context:
        return ""
    return (
        aggregate_raw_outputs_from_task_outputs(task_outputs)
        if task.context is NOT_SPECIFIED
        else aggregate_raw_outputs_from_tasks(task.context)
    )
```

- **`NOT_SPECIFIED`（默认）**：聚合所有前置任务输出——链式传递行为
- **`list[Task]`（用户指定）**：只取指定任务的输出——精确依赖
- **`None`（显式置空）**：不接收上下文

---

## 5.6 Callback 系统

Crew 提供四层回调机制：

```python
before_kickoff_callbacks: list[Callable[[dict | None], dict | None]]  # 执行前修改 inputs
after_kickoff_callbacks: list[Callable[[CrewOutput], CrewOutput]]     # 执行后修改结果
task_callback: Any | None   # 每个任务完成后触发
step_callback: Any | None   # 每个 Agent 推理步骤后触发
```

`before_kickoff_callbacks` 和 `after_kickoff_callbacks` 都是列表，支持链式调用：

```python
for before_callback in crew.before_kickoff_callbacks:
    normalized = before_callback(normalized)
for after_callback in self.after_kickoff_callbacks:
    result = after_callback(result)
```

`task_callback` 在初始化阶段分发到每个 Task，`step_callback` 通过 `setup_agents()` 分发到各 Agent。

---

## 5.7 kickoff 变体

**`kickoff_for_each()`**：批量执行，为每个输入创建 Crew 深拷贝：

```python
def kickoff_for_each(self, inputs: list[dict[str, Any]], ...) -> list[CrewOutput | CrewStreamingOutput]:
    results = []
    for input_data in inputs:
        crew = self.copy()
        output = crew.kickoff(inputs=input_data, input_files=input_files)
        results.append(output)
    return results
```

**`kickoff_async()`**：将 `kickoff()` 包装在 `asyncio.to_thread()` 中。

**`akickoff()`**：原生异步实现，使用 `asyncio.create_task` 替代线程池，内部调用 `_aexecute_tasks()` 执行任务，效率更高。

---

## 5.8 工具准备与注入

`_prepare_tools()` 在每个任务执行前动态构建工具列表：

```python
def _prepare_tools(self, agent: BaseAgent, task: Task, tools: list[BaseTool]) -> list[BaseTool]:
    # 1. 委派工具（hierarchical 用 _update_manager_tools，sequential 用 _add_delegation_tools）
    # 2. 代码执行工具
    # 3. 多模态工具（当 LLM 不原生支持时）
    # 4. Platform 工具（apps 集成）
    # 5. MCP 工具
    # 6. Memory 工具（recall / remember）
    # 7. 文件读取工具
    return tools
```

工具去重通过 `_merge_tools()` 静态方法实现，以工具名称为唯一标识。

---

## 5.9 验证器体系

Crew 使用大量 Pydantic 验证器在实例化阶段确保配置正确：

| 验证器 | 作用 |
|--------|------|
| `check_manager_llm` | hierarchical 必须有 manager |
| `validate_tasks` | sequential 模式 task 必须有 agent |
| `validate_end_with_at_most_one_async_task` | 末尾最多一个异步任务 |
| `validate_must_have_non_conditional_task` | 至少一个非条件任务 |
| `validate_first_task` | 第一个任务不能是 ConditionalTask |
| `validate_async_tasks_not_async` | ConditionalTask 不能异步 |
| `validate_context_no_future_tasks` | context 不能引用后续任务 |

---

## 5.10 其他功能

**`replay()`**：从指定任务重新执行，从 `TaskOutputStorageHandler` 恢复前置输出。

**`train()`**：强制 `human_input`、禁用 delegation，多次执行收集反馈，用 `TaskEvaluator` 生成训练数据。

**Memory 初始化**：`True` 用默认 Memory、传实例用自定义、`False` 禁用。

**Planning**：启用后 `CrewPlanner` 为每个任务生成执行计划，追加到任务描述中。

---

## 本章要点

- **Crew 是核心编排器**，将 Agent 和 Task 组织成协作团队，管理完整执行生命周期
- **两种 Process 模式**：`sequential`（顺序执行，Task 预先绑定 Agent）和 `hierarchical`（Manager Agent 通过 `AgentTools` 委派工具动态分配任务）
- **`kickoff()` 经历完整流程**：prepare_kickoff（回调、事件、插值、agent 初始化、planning）、按 Process 分发、after 回调、metrics 计算
- **`_execute_tasks()` 统一执行引擎**：支持同步/异步混合执行、ConditionalTask 条件跳过、replay 重放
- **上下文传递三种模式**：默认聚合所有前置输出、显式指定依赖任务、设为 `None` 不接收
- **四层回调**：before_kickoff、after_kickoff、task_callback、step_callback
- **`kickoff_for_each()` 通过深拷贝 Crew 实现批量执行**，`akickoff()` 提供原生异步执行
- **丰富的验证器体系**确保配置正确性，包括 agent 绑定、异步约束、context 依赖方向等检查
