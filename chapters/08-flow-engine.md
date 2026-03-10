# 第 8 章　Flow 执行引擎

上一章剖析了 Flow 的声明层——装饰器如何收集元数据，元类如何编译触发关系图。本章转向执行层：当调用 `flow.kickoff()` 时，引擎如何按图驱动方法执行、如何处理并发与竞争、如何在 sync 和 async 之间无缝切换，以及如何应对错误和 human-in-the-loop 场景。

Flow 执行引擎的全部逻辑集中在 `flow/flow.py` 中的 `Flow` 类，核心方法链为：`kickoff()` → `kickoff_async()` → `_execute_start_method()` → `_execute_method()` → `_execute_listeners()` → `_execute_single_listener()`。这是一个递归驱动的事件传播模型。

## 8.1 执行入口：kickoff() 与 kickoff_async()

`kickoff()` 是 Flow 的同步入口。它的实现揭示了 CrewAI 如何在同步和异步之间架桥：

```python
# crewai/flow/flow.py

def kickoff(
    self,
    inputs: dict[str, Any] | None = None,
    input_files: dict[str, FileInput] | None = None,
) -> Any | FlowStreamingOutput:
    # ... streaming 处理省略 ...

    async def _run_flow() -> Any:
        return await self.kickoff_async(inputs, input_files)

    try:
        asyncio.get_running_loop()
        # 已在 async 上下文中：用 ThreadPoolExecutor 隔离
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run_flow()).result()
    except RuntimeError:
        # 不在 async 上下文中：直接 asyncio.run
        return asyncio.run(_run_flow())
```

这段代码的策略是：

1. **无 event loop**：直接 `asyncio.run()` 启动异步执行。
2. **已有 event loop**（比如在 Jupyter 或其他 async 框架中）：在独立线程的 `ThreadPoolExecutor` 中运行 `asyncio.run()`，避免嵌套 event loop 的问题。

真正的执行逻辑在 `kickoff_async()` 中：

```python
async def kickoff_async(
    self,
    inputs: dict[str, Any] | None = None,
    input_files: dict[str, FileInput] | None = None,
) -> Any | FlowStreamingOutput:
    # 设置 OpenTelemetry baggage
    ctx = baggage.set_baggage("flow_inputs", inputs or {})
    flow_token = attach(ctx)

    # 设置 context variables
    flow_id_token = None
    request_id_token = None
    if current_flow_id.get() is None:
        flow_id_token = current_flow_id.set(self.flow_id)
    if current_flow_request_id.get() is None:
        request_id_token = current_flow_request_id.set(self.flow_id)

    try:
        # 重置执行状态（除非从持久化恢复）
        is_restoring = inputs and "id" in inputs and self._persistence is not None
        if not is_restoring:
            self._completed_methods.clear()
            self._method_outputs.clear()
            self._pending_and_listeners.clear()
            self._clear_or_listeners()
            self._method_call_counts.clear()

        # 处理输入和状态恢复
        if inputs:
            if "id" in inputs and self._persistence is not None:
                stored_state = self._persistence.load_state(inputs["id"])
                if stored_state:
                    self._restore_state(stored_state)
            filtered_inputs = {k: v for k, v in inputs.items() if k != "id"}
            if filtered_inputs:
                self._initialize_state(filtered_inputs)

        # 发射 FlowStartedEvent
        crewai_event_bus.emit(self, FlowStartedEvent(...))

        # 执行所有 start 方法
        unconditional_starts = [
            start_method
            for start_method in self._start_methods
            if not getattr(
                self._methods.get(start_method), "__trigger_methods__", None
            )
        ]
        starts_to_execute = (
            unconditional_starts if unconditional_starts else self._start_methods
        )
        tasks = [
            self._execute_start_method(start_method)
            for start_method in starts_to_execute
        ]
        await asyncio.gather(*tasks)

        # 获取最终输出
        final_output = self._method_outputs[-1] if self._method_outputs else None

        # 发射 FlowFinishedEvent
        crewai_event_bus.emit(self, FlowFinishedEvent(...))

        return final_output
    finally:
        # 清理 context tokens
        if request_id_token is not None:
            current_flow_request_id.reset(request_id_token)
        if flow_id_token is not None:
            current_flow_id.reset(flow_id_token)
        detach(flow_token)
```

关键设计决策值得逐一分析。

### 8.1.1 无条件 start 与条件 start 的区分

不是所有 `@start()` 方法都在 `kickoff()` 时立即执行。带有 `condition` 参数的 start 方法（如 `@start("approved")`）只有在其条件满足时才触发：

```python
unconditional_starts = [
    start_method
    for start_method in self._start_methods
    if not getattr(
        self._methods.get(start_method), "__trigger_methods__", None
    )
]
starts_to_execute = (
    unconditional_starts if unconditional_starts else self._start_methods
)
```

如果 Flow 中没有无条件 start（即所有 start 都是条件触发的），则回退为执行全部 start 方法——这是一个合理的兜底策略，避免 Flow 无法启动。

### 8.1.2 状态恢复机制

当 `inputs` 中包含 `"id"` 且配置了 persistence 时，引擎会尝试从存储中恢复之前的执行状态：

```python
if "id" in inputs and self._persistence is not None:
    restore_uuid = inputs["id"]
    stored_state = self._persistence.load_state(restore_uuid)
    if stored_state:
        self._restore_state(stored_state)
```

恢复时还会设置 `_is_execution_resuming` 标志，使引擎在遇到已完成的方法时跳过执行但继续传播 listener 链：

```python
if not is_restoring:
    self._completed_methods.clear()
    # ...
else:
    if self._completed_methods:
        self._is_execution_resuming = True
```

## 8.2 Start 方法执行

`_execute_start_method()` 是 start 方法的执行器，也是递归执行链的起点：

```python
async def _execute_start_method(self, start_method_name: FlowMethodName) -> None:
    # 处理已完成方法（恢复或循环场景）
    if start_method_name in self._completed_methods:
        if self._is_execution_resuming:
            last_output = self._method_outputs[-1] if self._method_outputs else None
            await self._execute_listeners(start_method_name, last_output)
            return
        # 循环 Flow：清除已完成标记，允许重新执行
        self._completed_methods.discard(start_method_name)
        self._clear_or_listeners()

    method = self._methods[start_method_name]
    enhanced_method = self._inject_trigger_payload_for_start_method(method)

    result, finished_event_id = await self._execute_method(
        start_method_name, enhanced_method
    )

    # 如果 start 方法同时是 router
    if start_method_name in self._routers and result is not None:
        await self._execute_listeners(start_method_name, result, finished_event_id)
        router_result_trigger = FlowMethodName(str(result))
        await self._execute_listeners(
            router_result_trigger, result, finished_event_id
        )
    else:
        await self._execute_listeners(start_method_name, result, finished_event_id)
```

几个关键点：

1. **恢复执行**：如果方法已标记为完成且正在恢复，跳过执行直接传播 listener——这让恢复的 Flow 能快速"追赶"到暂停点。
2. **循环 Flow**：如果方法已完成但不在恢复模式，说明这是一个循环执行——清除完成标记和 OR listener 状态，允许重新执行。
3. **Trigger Payload 注入**：对 start 方法特殊处理，将 `kickoff()` 的输入中的 `crewai_trigger_payload` 注入到方法参数中。
4. **Start + Router**：start 方法可以同时是 router，此时先触发监听方法名的 listener，再触发监听返回值的 listener。

## 8.3 方法执行核心：_execute_method()

`_execute_method()` 是所有方法执行的统一入口，它处理 sync/async 混合执行、事件发射和错误处理：

```python
async def _execute_method(
    self,
    method_name: FlowMethodName,
    method: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, str | None]:
    try:
        # 发射 MethodExecutionStartedEvent
        if not self.suppress_flow_events:
            crewai_event_bus.emit(self, MethodExecutionStartedEvent(...))

        # 设置方法名上下文
        from crewai.flow.flow_context import current_flow_method_name
        method_name_token = current_flow_method_name.set(method_name)

        try:
            if asyncio.iscoroutinefunction(method):
                result = await method(*args, **kwargs)
            else:
                # 同步方法在线程池中执行
                import contextvars
                ctx = contextvars.copy_context()
                result = await asyncio.to_thread(ctx.run, method, *args, **kwargs)
        finally:
            current_flow_method_name.reset(method_name_token)

        # 自动 await 从同步方法返回的 coroutine
        if asyncio.iscoroutine(result):
            result = await result

        self._method_outputs.append(result)
        self._method_execution_counts[method_name] = (
            self._method_execution_counts.get(method_name, 0) + 1
        )
        self._completed_methods.add(method_name)

        # 发射 MethodExecutionFinishedEvent
        finished_event_id = None
        if not self.suppress_flow_events:
            finished_event = MethodExecutionFinishedEvent(...)
            finished_event_id = finished_event.event_id
            crewai_event_bus.emit(self, finished_event)

        return result, finished_event_id

    except Exception as e:
        from crewai.flow.async_feedback.types import HumanFeedbackPending
        if isinstance(e, HumanFeedbackPending):
            e.context.method_name = method_name
            # ... 保存暂停状态
        elif not self.suppress_flow_events:
            crewai_event_bus.emit(self, MethodExecutionFailedEvent(...))
        raise e
```

### 8.3.1 Sync/Async 混合执行模型

这是 Flow 执行引擎最精巧的设计之一。开发者可以在同一个 Flow 中混用同步和异步方法，引擎自动处理差异：

```python
if asyncio.iscoroutinefunction(method):
    result = await method(*args, **kwargs)
else:
    import contextvars
    ctx = contextvars.copy_context()
    result = await asyncio.to_thread(ctx.run, method, *args, **kwargs)
```

对于 **async 方法**，直接 `await`。对于 **sync 方法**，使用 `asyncio.to_thread()` 将其提交到默认的 `ThreadPoolExecutor`。关键细节：

1. **`contextvars.copy_context()`**：在提交到线程前复制当前的 context variables，确保 `current_flow_id`、`current_flow_method_name` 等上下文在工作线程中可用。
2. **`ctx.run(method, *args, **kwargs)`**：在复制的 context 中执行方法，变量的修改不会泄漏到其他线程。
3. **自动 await 返回的 coroutine**：如果同步方法返回了一个 coroutine 对象（比如调用了但忘记 `await` 的异步函数），引擎会自动 `await` 它。

这种设计让 Flow 方法中可以直接调用 `Agent.kickoff()`（同步方法）或 `Crew.kickoff_for_each()`，而不会阻塞 event loop。

### 8.3.2 事件追踪链

每次方法执行都会发射一对事件：`MethodExecutionStartedEvent` 和 `MethodExecutionFinishedEvent`。Finished 事件携带一个 `event_id`，这个 ID 会被传递给下游的 listener，形成因果链（causal chain）：

```python
finished_event = MethodExecutionFinishedEvent(
    type="method_execution_finished",
    method_name=method_name,
    flow_name=self.name or self.__class__.__name__,
    state=self._copy_and_serialize_state(),
    result=result,
)
finished_event_id = finished_event.event_id
```

下游 listener 在 `triggered_by_scope` 中执行，将 `triggering_event_id` 记录到事件上下文中，便于 tracing 系统重建完整的执行调用链。

## 8.4 Listener 触发引擎

`_execute_listeners()` 是执行引擎的调度中枢。当一个方法完成后，它负责找到所有应该触发的下游方法并执行它们：

```python
async def _execute_listeners(
    self,
    trigger_method: FlowMethodName,
    result: Any,
    triggering_event_id: str | None = None,
) -> None:
    # 第一阶段：处理 router 链
    router_results = []
    current_trigger = trigger_method
    current_result = result

    while True:
        routers_triggered = self._find_triggered_methods(
            current_trigger, router_only=True
        )
        if not routers_triggered:
            break

        for router_name in routers_triggered:
            router_result, current_triggering_event_id = \
                await self._execute_single_listener(
                    router_name, current_result, current_triggering_event_id
                )
            if router_result:
                router_result_str = (
                    router_result.value
                    if isinstance(router_result, enum.Enum)
                    else str(router_result)
                )
                router_results.append(FlowMethodName(router_result_str))
                current_trigger = FlowMethodName(router_result_str)

    # 第二阶段：执行普通 listener（并行）
    all_triggers = [trigger_method, *router_results]

    for current_trigger in all_triggers:
        if current_trigger:
            listeners_triggered = self._find_triggered_methods(
                current_trigger, router_only=False
            )
            if listeners_triggered:
                # 检查是否存在竞争组
                racing_group = self._get_racing_group_for_listeners(
                    listeners_triggered
                )
                if racing_group:
                    racing_members, _ = racing_group
                    other_listeners = [
                        name for name in listeners_triggered
                        if name not in racing_members
                    ]
                    await self._execute_racing_listeners(
                        racing_members, other_listeners,
                        result, current_triggering_event_id,
                    )
                else:
                    tasks = [
                        self._execute_single_listener(
                            listener_name, result,
                            current_triggering_event_id,
                        )
                        for listener_name in listeners_triggered
                    ]
                    await asyncio.gather(*tasks)
```

执行策略分为两个阶段：

**第一阶段——Router 链**：顺序执行所有被触发的 router。Router 是决策节点，其返回值决定了后续路径，因此必须顺序执行。注意 `while True` 循环——router 可以链式触发其他 router，形成决策树。

**第二阶段——Listener 并行执行**：对原始触发方法和所有 router 结果，分别查找并触发对应的 listener。同一批 listener 通过 `asyncio.gather()` 并行执行。

### 8.4.1 触发方法的查找逻辑

`_find_triggered_methods()` 是条件评估的核心：

```python
def _find_triggered_methods(
    self, trigger_method: FlowMethodName, router_only: bool
) -> list[FlowMethodName]:
    triggered: list[FlowMethodName] = []

    for listener_name, condition_data in self._listeners.items():
        is_router = listener_name in self._routers

        # 根据 router_only 过滤
        if router_only != is_router:
            continue

        # 跳过条件 start 方法（它们由专门的逻辑处理）
        if not router_only and listener_name in self._start_methods:
            continue

        if is_simple_flow_condition(condition_data):
            condition_type, methods = condition_data

            if condition_type == OR_CONDITION:
                has_multiple_triggers = len(methods) > 1
                should_check_fired = has_multiple_triggers and not is_router

                if (
                    not should_check_fired
                    or listener_name not in self._fired_or_listeners
                ):
                    if trigger_method in methods:
                        triggered.append(listener_name)
                        if should_check_fired:
                            self._fired_or_listeners.add(listener_name)

            elif condition_type == AND_CONDITION:
                pending_key = PendingListenerKey(listener_name)
                if pending_key not in self._pending_and_listeners:
                    self._pending_and_listeners[pending_key] = set(methods)
                if trigger_method in self._pending_and_listeners[pending_key]:
                    self._pending_and_listeners[pending_key].discard(trigger_method)

                if not self._pending_and_listeners[pending_key]:
                    triggered.append(listener_name)
                    self._pending_and_listeners.pop(pending_key, None)

        elif is_flow_condition_dict(condition_data):
            # 复合条件：递归评估
            if self._evaluate_condition(
                condition_data, trigger_method, listener_name
            ):
                triggered.append(listener_name)

    return triggered
```

这段代码体现了几个重要的语义规则：

1. **Router 和 Listener 分开评估**：通过 `router_only` 参数控制，确保 router 先于 listener 执行。
2. **OR 条件的一次性语义**：多源 OR listener（如 `or_("a", "b")`）只在第一个匹配的触发方法完成时执行一次。单源 listener 和 router 不受此限制。
3. **AND 条件的累积语义**：`_pending_and_listeners` 维护待完成方法集合，每次触发方法完成时从集合中移除，集合为空时触发 listener。
4. **条件 start 方法被跳过**：条件 start 方法有专门的处理逻辑（在 `_execute_listeners` 的末尾），不参与普通 listener 的查找。

### 8.4.2 OR 竞争语义与 Racing Groups

当多个方法同时作为 OR listener 的触发源时，它们之间存在"竞争"关系——只有最先完成的才能触发 listener。CrewAI 通过 Racing Groups 机制精确处理这种场景：

```python
def _build_racing_groups(self) -> dict[frozenset[FlowMethodName], FlowMethodName]:
    """Identify groups of methods that race for the same OR listener."""
    racing_groups: dict[frozenset[FlowMethodName], FlowMethodName] = {}

    # 构建方法到 listener 的反向映射
    method_to_listeners: dict[FlowMethodName, set[FlowMethodName]] = {}
    for listener_name, condition_data in self._listeners.items():
        if is_simple_flow_condition(condition_data):
            _, methods = condition_data
            for m in methods:
                method_to_listeners.setdefault(m, set()).add(listener_name)

    # 查找排他性竞争组
    for listener_name, condition_data in self._listeners.items():
        if listener_name in self._routers:
            continue  # Router 不参与竞争

        trigger_methods: set[FlowMethodName] = set()
        if is_simple_flow_condition(condition_data):
            condition_type, methods = condition_data
            if condition_type == OR_CONDITION and len(methods) > 1:
                trigger_methods = set(methods)

        if trigger_methods:
            # 只有排他性方法才加入竞争组
            exclusive_methods = {
                m for m in trigger_methods
                if method_to_listeners.get(m, set()) == {listener_name}
            }
            if len(exclusive_methods) > 1:
                racing_groups[frozenset(exclusive_methods)] = listener_name

    return racing_groups
```

"排他性"的含义是：只有那些**仅被**这个 OR listener 监听的方法才参与竞争。如果一个方法同时是其他 listener 的触发源，它不会被取消——因为取消它可能影响其他 listener 的执行。

竞争执行通过 `asyncio.as_completed()` 实现先到先得：

```python
async def _execute_racing_listeners(
    self,
    racing_listeners: frozenset[FlowMethodName],
    other_listeners: list[FlowMethodName],
    result: Any,
    triggering_event_id: str | None = None,
) -> None:
    racing_tasks = [
        asyncio.create_task(
            self._execute_single_listener(name, result, triggering_event_id),
            name=str(name),
        )
        for name in racing_listeners
    ]

    other_tasks = [
        asyncio.create_task(
            self._execute_single_listener(name, result, triggering_event_id),
            name=str(name),
        )
        for name in other_listeners
    ]

    if racing_tasks:
        for coro in asyncio.as_completed(racing_tasks):
            try:
                await coro
            except Exception as e:
                logger.debug(f"Racing listener failed: {e}")
                continue
            break  # 第一个成功完成的赢得竞争

        # 取消其他竞争任务
        for task in racing_tasks:
            if not task.done():
                task.cancel()

    # 非竞争 listener 正常并行执行
    if other_tasks:
        await asyncio.gather(*other_tasks, return_exceptions=True)
```

这个设计确保了 OR 语义的精确实现：第一个完成的方法触发 listener，其他方法被取消。

## 8.5 单个 Listener 的执行

`_execute_single_listener()` 处理单个 listener 的完整生命周期：

```python
async def _execute_single_listener(
    self,
    listener_name: FlowMethodName,
    result: Any,
    triggering_event_id: str | None = None,
) -> tuple[Any, str | None]:
    # 递归调用保护
    count = self._method_call_counts.get(listener_name, 0) + 1
    if count > self._max_method_calls:
        raise RecursionError(
            f"Method '{listener_name}' has been called {self._max_method_calls} times "
            f"in this flow execution, which indicates an infinite loop."
        )
    self._method_call_counts[listener_name] = count

    # 处理已完成方法
    if listener_name in self._completed_methods:
        if self._is_execution_resuming:
            await self._execute_listeners(listener_name, None)
            return (None, None)
        # 循环 Flow：清除标记允许重新执行
        self._completed_methods.discard(listener_name)
        self._clear_or_listeners()

    try:
        method = self._methods[listener_name]

        # 检查方法签名以决定是否传递上游结果
        sig = inspect.signature(method)
        params = list(sig.parameters.values())
        method_params = [p for p in params if p.name != "self"]

        if triggering_event_id:
            with triggered_by_scope(triggering_event_id):
                if method_params:
                    listener_result, finished_event_id = await self._execute_method(
                        listener_name, method, result
                    )
                else:
                    listener_result, finished_event_id = await self._execute_method(
                        listener_name, method
                    )
        else:
            if method_params:
                listener_result, finished_event_id = await self._execute_method(
                    listener_name, method, result
                )
            else:
                listener_result, finished_event_id = await self._execute_method(
                    listener_name, method
                )

        # 递归触发下游 listener
        await self._execute_listeners(
            listener_name, listener_result, finished_event_id
        )

        return (listener_result, finished_event_id)

    except Exception as e:
        from crewai.flow.async_feedback.types import HumanFeedbackPending
        if not isinstance(e, HumanFeedbackPending):
            logger.error(f"Error executing listener {listener_name}: {e}")
        raise
```

### 8.5.1 递归保护机制

`_method_call_counts` 跟踪每个方法在本次 Flow 执行中被调用的次数。超过 `_max_method_calls`（默认 100）时抛出 `RecursionError`：

```python
count = self._method_call_counts.get(listener_name, 0) + 1
if count > self._max_method_calls:
    raise RecursionError(
        f"Method '{listener_name}' has been called {self._max_method_calls} times..."
    )
```

这个保护在循环 Flow 中尤为重要——比如 `method_a` → `method_b` → `method_a` 形成的循环，如果没有终止条件，会永远执行下去。

### 8.5.2 参数注入的签名检查

引擎通过 `inspect.signature()` 检查 listener 的参数签名，决定是否传递上游方法的结果：

```python
sig = inspect.signature(method)
params = list(sig.parameters.values())
method_params = [p for p in params if p.name != "self"]

if method_params:
    listener_result, _ = await self._execute_method(listener_name, method, result)
else:
    listener_result, _ = await self._execute_method(listener_name, method)
```

这意味着开发者可以选择性地接收上游数据：

```python
@listen("fetch_data")
def with_data(self, data):  # 接收上游返回值
    pass

@listen("fetch_data")
def without_data(self):  # 不关心上游返回值
    pass
```

### 8.5.3 递归传播

每个 listener 执行完成后，它自身也成为一个触发源，递归调用 `_execute_listeners()`：

```python
await self._execute_listeners(
    listener_name, listener_result, finished_event_id
)
```

这形成了一个深度优先的事件传播链：A → B → C → D。每个节点完成后立即触发其下游。多个并行 listener 通过 `asyncio.gather()` 并发执行，但每条路径内部是顺序的。

## 8.6 DAG 分析与拓扑工具

虽然 Flow 的执行引擎采用递归事件驱动而非显式的拓扑排序，但 `flow/utils.py` 中提供了丰富的图分析工具，主要用于可视化和调试。

### 8.6.1 层级计算

`calculate_node_levels()` 对 Flow 的方法图进行 BFS 层级计算：

```python
# crewai/flow/utils.py

def calculate_node_levels(flow: Any) -> dict[str, int]:
    levels: dict[str, int] = {}
    queue: deque[str] = deque()
    visited: set[str] = set()
    pending_and_listeners: dict[str, set[str]] = {}

    # start 方法为第 0 层
    for method_name, method in flow._methods.items():
        if hasattr(method, "__is_start_method__"):
            levels[method_name] = 0
            queue.append(method_name)

    # 预计算 listener 依赖
    or_listeners = defaultdict(list)
    and_listeners = defaultdict(set)
    for listener_name, condition_data in flow._listeners.items():
        if isinstance(condition_data, tuple):
            condition_type, trigger_methods = condition_data
        elif isinstance(condition_data, dict):
            trigger_methods = _extract_all_methods_recursive(condition_data, flow)
            condition_type = condition_data.get("type", "OR")
        else:
            continue

        if condition_type == "OR":
            for method in trigger_methods:
                or_listeners[method].append(listener_name)
        elif condition_type == "AND":
            and_listeners[listener_name] = set(trigger_methods)

    # BFS 遍历
    while queue:
        current = queue.popleft()
        current_level = levels[current]
        visited.add(current)

        # OR listener 的层级 = 触发方法层级 + 1
        for listener_name in or_listeners[current]:
            if listener_name not in levels or levels[listener_name] > current_level + 1:
                levels[listener_name] = current_level + 1
                if listener_name not in visited:
                    queue.append(listener_name)

        # AND listener 需要所有触发方法都已分配层级
        for listener_name, required_methods in and_listeners.items():
            if current in required_methods:
                if listener_name not in pending_and_listeners:
                    pending_and_listeners[listener_name] = set()
                pending_and_listeners[listener_name].add(current)

                if required_methods == pending_and_listeners[listener_name]:
                    if (
                        listener_name not in levels
                        or levels[listener_name] > current_level + 1
                    ):
                        levels[listener_name] = current_level + 1
                        if listener_name not in visited:
                            queue.append(listener_name)

        # 处理 router 路径
        process_router_paths(flow, current, current_level, levels, queue)

    return levels
```

这个 BFS 算法为每个方法分配层级，规则是：
- Start 方法在第 0 层
- OR listener 在其任一触发方法的层级 + 1
- AND listener 在其所有触发方法都就绪后，层级为最后一个就绪方法的层级 + 1
- Router 的路径目标在 router 的层级 + 1

### 8.6.2 祖先关系与父子关系

`build_ancestor_dict()` 构建每个节点的祖先集合，`build_parent_children_dict()` 构建父子关系映射：

```python
def build_ancestor_dict(flow: Any) -> dict[str, set[str]]:
    ancestors: dict[str, set[str]] = {node: set() for node in flow._methods}
    visited: set[str] = set()
    for node in flow._methods:
        if node not in visited:
            dfs_ancestors(node, ancestors, visited, flow)
    return ancestors
```

这些工具函数在 Flow 可视化（`flow.plot()`）中被使用，用于正确布局节点位置和绘制依赖关系边。

## 8.7 Human-in-the-Loop：暂停与恢复

Flow 引擎支持在执行过程中暂停等待人工反馈，然后恢复执行。这通过异常控制流实现：

```python
# 在 _execute_method 中
except Exception as e:
    from crewai.flow.async_feedback.types import HumanFeedbackPending
    if isinstance(e, HumanFeedbackPending):
        e.context.method_name = method_name
        # 保存暂停状态到 persistence
        if self._persistence is None:
            from crewai.flow.persistence import SQLiteFlowPersistence
            self._persistence = SQLiteFlowPersistence()
        # 发射暂停事件
        crewai_event_bus.emit(self, MethodExecutionPausedEvent(...))
    raise e
```

`HumanFeedbackPending` 是一种特殊的异常，它携带暂停上下文（包括 flow ID、方法名、消息等），沿调用栈向上传播直到 `kickoff_async()` 捕获它：

```python
# kickoff_async 中
except Exception as e:
    from crewai.flow.async_feedback.types import HumanFeedbackPending
    if isinstance(e, HumanFeedbackPending):
        # 保存状态和暂停上下文
        self._persistence.save_pending_feedback(
            flow_uuid=e.context.flow_id,
            context=e.context,
            state_data=state_data,
        )
        # 发射 FlowPausedEvent
        crewai_event_bus.emit(self, FlowPausedEvent(...))
        return e  # 返回而非抛出
```

恢复执行通过 `from_pending()` 类方法和 `resume()` / `resume_async()` 方法：

```python
@classmethod
def from_pending(
    cls,
    flow_id: str,
    persistence: FlowPersistence | None = None,
    **kwargs: Any,
) -> Flow[Any]:
    if persistence is None:
        from crewai.flow.persistence import SQLiteFlowPersistence
        persistence = SQLiteFlowPersistence()

    loaded = persistence.load_pending_feedback(flow_id)
    if loaded is None:
        raise ValueError(f"No pending feedback found for flow_id: {flow_id}")

    state_data, pending_context = loaded

    instance = cls(persistence=persistence, **kwargs)
    instance._initialize_state(state_data)
    instance._pending_feedback_context = pending_context
    instance._is_execution_resuming = True
    instance._completed_methods.add(FlowMethodName(pending_context.method_name))

    return instance
```

典型的 human-in-the-loop 工作流：

```python
# 第一次执行——暂停等待反馈
flow = MyReviewFlow(persistence=SQLiteFlowPersistence())
result = flow.kickoff()
# result 是 HumanFeedbackPending 对象

# 稍后恢复——提供人工反馈
flow = MyReviewFlow.from_pending("flow-uuid-here")
result = flow.resume("Looks good, approved!")
```

## 8.8 self.ask()：同步的人工输入

除了基于异常的暂停/恢复模式，Flow 还提供了更简单的 `self.ask()` 方法，用于在方法执行中同步请求人工输入：

```python
def ask(
    self,
    message: str,
    timeout: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    # 发射输入请求事件
    crewai_event_bus.emit(self, FlowInputRequestedEvent(...))

    # 自动 checkpoint 状态
    self._checkpoint_state_for_ask()

    # 解析 InputProvider
    provider = self._resolve_input_provider()

    if timeout is not None:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(provider.request_input, message, self, metadata)
        try:
            raw = future.result(timeout=timeout)
        except FuturesTimeoutError:
            future.cancel()
            raw = None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    else:
        raw = provider.request_input(message, self, metadata=metadata)

    # 记录历史
    self._input_history.append({...})
    # 发射输入接收事件
    crewai_event_bus.emit(self, FlowInputReceivedEvent(...))

    return response
```

`ask()` 的 InputProvider 解析遵循优先级链：

1. 实例级 `self.input_provider`
2. 全局配置 `flow_config.input_provider`
3. 内置的 `ConsoleProvider`（标准输入）

`timeout` 参数通过 `ThreadPoolExecutor` + `Future.result(timeout=...)` 实现。超时后返回 `None`，让开发者可以优雅地处理超时场景。

## 8.9 循环 Flow 的执行语义

Flow 支持循环执行——方法 A 触发方法 B，方法 B 又触发方法 A。这在迭代优化场景中非常常见：

```python
class IterativeFlow(Flow):
    @start()
    def generate(self):
        return self.state.draft

    @listen("generate")
    def evaluate(self, draft):
        self.state.score = score_draft(draft)
        return self.state.score

    @router("evaluate")
    def decide(self, score):
        if score > 0.8:
            return "DONE"
        return "RETRY"

    @listen("RETRY")
    def improve(self):
        self.state.draft = improve_draft(self.state.draft)
        return self.state.draft

    @listen("improve")
    def re_evaluate(self, draft):
        # 触发 evaluate，形成循环
        ...
```

循环执行的关键机制：

1. **清除完成标记**：当一个已完成的方法需要再次执行时，引擎从 `_completed_methods` 中移除它。
2. **清除 OR listener 状态**：`_clear_or_listeners()` 重置所有 OR listener 的触发记录，允许它们在新一轮循环中再次触发。
3. **递归保护**：`_max_method_calls`（默认 100）防止无终止条件的循环导致栈溢出。

```python
if listener_name in self._completed_methods:
    if self._is_execution_resuming:
        await self._execute_listeners(listener_name, None)
        return (None, None)
    # 循环 Flow：清除标记，允许重新执行
    self._completed_methods.discard(listener_name)
    self._clear_or_listeners()
```

## 8.10 错误处理策略

Flow 引擎的错误处理遵循"让它传播"的策略。方法执行中的异常会沿调用栈向上传播，但有几种特殊处理：

1. **HumanFeedbackPending**：不是错误，是暂停信号。捕获后保存状态并返回。
2. **RecursionError**：由引擎自身抛出，表示循环检测触发。
3. **普通异常**：发射 `MethodExecutionFailedEvent` 事件后重新抛出。

```python
except Exception as e:
    from crewai.flow.async_feedback.types import HumanFeedbackPending
    if isinstance(e, HumanFeedbackPending):
        # 暂停，不是错误
        e.context.method_name = method_name
    elif not self.suppress_flow_events:
        crewai_event_bus.emit(self, MethodExecutionFailedEvent(
            type="method_execution_failed",
            method_name=method_name,
            flow_name=self.name or self.__class__.__name__,
            error=e,
        ))
    raise e
```

在 `_execute_single_listener()` 层面：

```python
except Exception as e:
    from crewai.flow.async_feedback.types import HumanFeedbackPending
    if not isinstance(e, HumanFeedbackPending):
        logger.error(f"Error executing listener {listener_name}: {e}")
    raise
```

这意味着：
- 单个 listener 的失败会终止整个 Flow 的执行（异常传播到 `kickoff_async()`）。
- 并行 listener 中，一个失败会通过 `asyncio.gather()` 被收集——但由于默认不使用 `return_exceptions=True`，第一个失败的 listener 会导致 `gather` 抛出异常。

一个值得注意的例外是 Racing Listeners 中的处理：

```python
if racing_tasks:
    for coro in asyncio.as_completed(racing_tasks):
        try:
            await coro
        except Exception as e:
            logger.debug(f"Racing listener failed: {e}")
            continue  # 失败不终止，等待下一个完成
        break
```

在竞争执行中，失败的 listener 不会终止竞争——引擎会继续等待下一个完成的 listener。只有当所有竞争 listener 都失败时，才算真正失败。

## 8.11 事件系统集成

Flow 执行引擎与 CrewAI 的事件总线深度集成。整个执行生命周期中发射的事件包括：

| 事件 | 时机 |
|------|------|
| `FlowCreatedEvent` | Flow 实例化时 |
| `FlowStartedEvent` | `kickoff_async()` 开始时 |
| `MethodExecutionStartedEvent` | 每个方法开始执行时 |
| `MethodExecutionFinishedEvent` | 每个方法执行完成时 |
| `MethodExecutionFailedEvent` | 方法执行抛出异常时 |
| `MethodExecutionPausedEvent` | 方法因 HumanFeedback 暂停时 |
| `FlowPausedEvent` | 整个 Flow 因 HumanFeedback 暂停时 |
| `FlowFinishedEvent` | Flow 执行完成时 |
| `FlowInputRequestedEvent` | `self.ask()` 请求输入时 |
| `FlowInputReceivedEvent` | `self.ask()` 收到输入时 |

事件通过 `crewai_event_bus.emit()` 发射，返回一个 `Future`。引擎在关键点等待这些 Future 完成：

```python
if self._event_futures:
    await asyncio.gather(
        *[asyncio.wrap_future(f) for f in self._event_futures]
    )
    self._event_futures.clear()
```

这确保了事件处理器（如 tracing listener）在 Flow 结束前有机会完成处理。

## 8.12 Streaming 支持

Flow 支持 streaming 模式，通过 `stream=True` 启用：

```python
def kickoff(self, inputs=None, input_files=None):
    if self.stream:
        result_holder: list[Any] = []
        state = create_streaming_state(...)
        output_holder = []

        def run_flow() -> None:
            try:
                self.stream = False  # 避免递归 streaming
                result = self.kickoff(inputs=inputs, input_files=input_files)
                result_holder.append(result)
            except Exception as e:
                signal_error(state, e)
            finally:
                self.stream = True
                signal_end(state)

        streaming_output = FlowStreamingOutput(
            sync_iterator=create_chunk_generator(state, run_flow, output_holder)
        )
        return streaming_output
```

Streaming 的实现策略是"包装"：在独立线程中执行普通的 `kickoff()`，通过 `create_chunk_generator` 生成器将中间输出实时传递给调用者。注意 `self.stream = False` 的临时修改——避免递归调用再次进入 streaming 分支。

## 本章要点

- **kickoff()** 是同步入口，通过 `asyncio.run()` 或 `ThreadPoolExecutor` + `asyncio.run()` 桥接到异步执行引擎 `kickoff_async()`。
- **kickoff_async()** 区分无条件 start 和条件 start；若全部都是条件 start 则回退为执行全部——确保 Flow 始终能启动。
- **Sync/Async 混合执行**：async 方法直接 `await`；sync 方法通过 `asyncio.to_thread()` + `contextvars.copy_context()` 在线程池中执行，保持上下文变量的正确传播。
- **执行传播模型**是递归事件驱动：方法完成 → `_execute_listeners()` 查找触发的下游方法 → `_execute_single_listener()` 执行 → 递归调用 `_execute_listeners()`。
- **Router 先于 Listener 执行**：`_execute_listeners()` 分两阶段——先顺序执行 router 链，再并行执行普通 listener；router 的返回值作为新的触发源。
- **AND 条件**通过 `_pending_and_listeners` 字典跟踪待完成集合；**OR 条件**通过 `_fired_or_listeners` 确保多源 listener 只触发一次。
- **Racing Groups** 机制通过 `asyncio.as_completed()` 实现 OR 竞争语义——第一个完成的 listener 赢得执行权，其余被取消。
- **循环 Flow** 通过清除 `_completed_methods` 和 `_fired_or_listeners` 实现重新执行，`_max_method_calls`（默认 100）作为安全阀防止无限循环。
- **Human-in-the-Loop** 通过 `HumanFeedbackPending` 异常实现暂停/恢复；`self.ask()` 提供同步的人工输入接口，支持 timeout 和可插拔的 InputProvider。
- **错误处理**遵循"传播"策略：普通异常终止 Flow 执行，`HumanFeedbackPending` 被特殊处理为暂停信号，Racing Listeners 中失败不终止竞争。
- **事件系统**贯穿整个执行生命周期，每个方法的开始、完成、失败、暂停都有对应事件，通过 `event_id` 形成因果追踪链。
