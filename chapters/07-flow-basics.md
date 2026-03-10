# 第 7 章　Flow 基础：装饰器模型

在前面的章节中，我们深入分析了 Agent、Task、Crew 的协作机制。然而，当业务逻辑变得更加复杂——需要条件分支、多步编排、状态传递——单纯的 Crew 已经不够用了。CrewAI 的 Flow 系统正是为此而生：它提供了一套基于装饰器的声明式编排框架，让开发者用 Python 方法定义工作流节点，用装饰器声明节点间的触发关系，最终形成一张有向图（DAG）来驱动执行。

本章聚焦于 Flow 系统的"声明层"——装饰器模型和状态管理。我们将从源码层面剖析 `@start()`、`@listen()`、`@router()` 三大装饰器的实现原理，以及 `FlowState` 的设计哲学。

## 7.1 整体架构概览

Flow 系统的核心文件分布如下：

| 文件 | 职责 |
|------|------|
| `flow/flow.py` | Flow 基类、装饰器函数、元类、执行引擎 |
| `flow/flow_wrappers.py` | FlowMethod 包装类（StartMethod、ListenMethod、RouterMethod） |
| `flow/types.py` | 类型定义（FlowMethodName、FlowState 等） |
| `flow/constants.py` | 常量定义（AND_CONDITION、OR_CONDITION） |
| `flow/utils.py` | 工具函数（条件解析、AST 分析等） |
| `flow/flow_context.py` | 运行时上下文变量 |

Flow 的设计遵循"声明与执行分离"的原则：装饰器在类定义阶段收集元数据，元类在类创建时构建触发关系图，执行引擎在 `kickoff()` 时按图驱动。这一分离让开发者只需关注"做什么"和"何时做"，而不必操心执行顺序和并发控制。

## 7.2 FlowState：有状态的工作流基座

每个 Flow 实例都拥有一份状态（state），在整个工作流执行过程中被所有方法共享和修改。`FlowState` 是状态的推荐基类：

```python
# crewai/flow/flow.py

class FlowState(BaseModel):
    """Base model for all flow states, ensuring each state has a unique ID."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the flow state",
    )
```

`FlowState` 继承自 Pydantic 的 `BaseModel`，只定义了一个 `id` 字段，使用 UUID 自动生成。这个设计有几个重要考量：

1. **唯一标识**：每个 Flow 执行实例都有唯一的 `id`，便于持久化和恢复。
2. **Pydantic 验证**：子类可以利用 Pydantic 的类型验证、序列化等能力。
3. **灵活扩展**：开发者只需继承 `FlowState` 并添加自定义字段。

典型用法如下：

```python
from crewai.flow.flow import Flow, FlowState, start, listen

class ResearchState(FlowState):
    topic: str = ""
    findings: list[str] = []
    summary: str = ""

class ResearchFlow(Flow[ResearchState]):
    @start()
    def begin(self):
        self.state.topic = "AI Safety"
        return self.state.topic

    @listen("begin")
    def research(self, topic):
        self.state.findings.append(f"Finding about {topic}")
        return self.state.findings
```

值得注意的是，Flow 也支持普通字典作为 state：

```python
class SimpleFlow(Flow[dict]):
    @start()
    def begin(self):
        self.state["count"] = 0
```

### 7.2.1 StateProxy：线程安全的状态代理

由于 Flow 中多个 listener 可能并行执行，直接操作 state 会导致竞态条件。CrewAI 通过 `StateProxy` 解决了这个问题：

```python
# crewai/flow/flow.py

class StateProxy(Generic[T]):
    """Proxy that provides thread-safe access to flow state."""

    __slots__ = ("_proxy_lock", "_proxy_state")

    def __init__(self, state: T, lock: threading.Lock) -> None:
        object.__setattr__(self, "_proxy_state", state)
        object.__setattr__(self, "_proxy_lock", lock)

    def __getattr__(self, name: str) -> Any:
        value = getattr(object.__getattribute__(self, "_proxy_state"), name)
        lock = object.__getattribute__(self, "_proxy_lock")
        if isinstance(value, list):
            return LockedListProxy(value, lock)
        if isinstance(value, dict):
            return LockedDictProxy(value, lock)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_proxy_state", "_proxy_lock"):
            object.__setattr__(self, name, value)
        else:
            with object.__getattribute__(self, "_proxy_lock"):
                setattr(object.__getattribute__(self, "_proxy_state"), name, value)
```

`StateProxy` 的精妙之处在于：

- **读操作**：直接返回底层值，对 `list` 和 `dict` 类型额外包装为 `LockedListProxy` / `LockedDictProxy`。
- **写操作**：所有赋值操作都通过 `_proxy_lock` 加锁保护。
- **透明代理**：开发者写 `self.state.topic = "AI"` 时感知不到锁的存在。

Flow 类通过 `@property` 返回 StateProxy 而非原始 state：

```python
@property
def state(self) -> T:
    return StateProxy(self._state, self._state_lock)
```

配套的 `LockedListProxy` 继承自 `list`，确保 `isinstance(proxy, list)` 返回 `True`，同时对所有写入操作加锁：

```python
class LockedListProxy(list, Generic[T]):
    def __init__(self, lst: list[T], lock: threading.Lock) -> None:
        super().__init__()
        self._list = lst
        self._lock = lock

    def append(self, item: T) -> None:
        with self._lock:
            self._list.append(item)

    def extend(self, items: Iterable[T]) -> None:
        with self._lock:
            self._list.extend(items)
    # ... 其他写操作同理加锁
```

这种设计在保持 API 兼容性的同时，确保了并行 listener 修改 state 时的数据一致性。

## 7.3 FlowMethod：装饰器的包装层

在深入三大装饰器之前，我们先看它们共用的底层包装类。所有被装饰的方法都会被包装为 `FlowMethod` 的子类：

```python
# crewai/flow/flow_wrappers.py

class FlowMethod(Generic[P, R]):
    """Base wrapper for flow methods with decorator metadata."""

    def __init__(self, meth: Callable[P, R], instance: Any = None) -> None:
        self._meth = meth
        self._instance = instance
        functools.update_wrapper(self, meth, updated=[])
        self.__name__: FlowMethodName = FlowMethodName(self.__name__)
        self.__signature__ = inspect.signature(meth)

        if instance is not None:
            self.__self__ = instance

        if inspect.iscoroutinefunction(meth):
            try:
                inspect.markcoroutinefunction(self)
            except AttributeError:
                import asyncio.coroutines
                self._is_coroutine = asyncio.coroutines._is_coroutine
```

`FlowMethod` 的核心职责：

1. **保留原始方法信息**：通过 `functools.update_wrapper` 复制 `__name__`、`__doc__` 等属性。
2. **描述符协议**：实现 `__get__` 方法，支持作为实例方法绑定。
3. **异步标记**：如果原始方法是 coroutine，包装后仍被正确识别为 coroutine。
4. **属性传递**：保留 `__is_router__`、`__router_paths__` 等流控属性。

三个子类分别标记不同的角色：

```python
class StartMethod(FlowMethod[P, R]):
    """Wrapper for methods marked as flow start points."""
    __is_start_method__: bool = True
    __trigger_methods__: list[FlowMethodName] | None = None
    __condition_type__: FlowConditionType | None = None
    __trigger_condition__: FlowCondition | None = None

class ListenMethod(FlowMethod[P, R]):
    """Wrapper for methods marked as flow listeners."""
    __trigger_methods__: list[FlowMethodName] | None = None
    __condition_type__: FlowConditionType | None = None
    __trigger_condition__: FlowCondition | None = None

class RouterMethod(FlowMethod[P, R]):
    """Wrapper for methods marked as flow routers."""
    __is_router__: bool = True
    __trigger_methods__: list[FlowMethodName] | None = None
    __condition_type__: FlowConditionType | None = None
    __trigger_condition__: FlowCondition | None = None
```

注意这三个子类共享几乎相同的属性结构，区别仅在于标志位：`__is_start_method__` 和 `__is_router__`。这意味着一个方法可以同时是 start 和 router（用于需要在入口就做条件路由的场景）。

## 7.4 @start() 装饰器：流程入口

`@start()` 标记一个方法为 Flow 的起始点。当调用 `flow.kickoff()` 时，所有 start 方法会被并行执行。

```python
# crewai/flow/flow.py

def start(
    condition: str | FlowCondition | Callable[..., Any] | None = None,
) -> Callable[[Callable[P, R]], StartMethod[P, R]]:
    """Marks a method as a flow's starting point."""

    def decorator(func: Callable[P, R]) -> StartMethod[P, R]:
        wrapper = StartMethod(func)

        if condition is not None:
            if is_flow_method_name(condition):
                wrapper.__trigger_methods__ = [condition]
                wrapper.__condition_type__ = OR_CONDITION
            elif is_flow_condition_dict(condition):
                if "conditions" in condition:
                    wrapper.__trigger_condition__ = condition
                    wrapper.__trigger_methods__ = _extract_all_methods(condition)
                    wrapper.__condition_type__ = condition["type"]
                elif "methods" in condition:
                    wrapper.__trigger_methods__ = condition["methods"]
                    wrapper.__condition_type__ = condition["type"]
                else:
                    raise ValueError(
                        "Condition dict must contain 'conditions' or 'methods'"
                    )
            elif is_flow_method_callable(condition):
                wrapper.__trigger_methods__ = [condition.__name__]
                wrapper.__condition_type__ = OR_CONDITION
            else:
                raise ValueError(
                    "Condition must be a method, string, or a result of or_() or and_()"
                )
        return wrapper

    return decorator
```

`@start()` 有两种使用模式：

**无条件启动**——最常见的用法，在 `kickoff()` 时自动执行：

```python
@start()
def begin(self):
    return "Hello, Flow!"
```

**条件启动**——只在指定方法完成后才触发，适用于 router 驱动的分支入口：

```python
@start("approved")
def process_approved(self):
    # 当某个 router 返回 "approved" 时触发
    pass
```

`start()` 函数本身是一个装饰器工厂——它返回一个 decorator，decorator 返回一个 `StartMethod` 包装器。条件参数 `condition` 支持四种形式：

| 形式 | 示例 | 含义 |
|------|------|------|
| `None` | `@start()` | 无条件启动 |
| `str` | `@start("method_a")` | 监听方法名 |
| `Callable` | `@start(method_a)` | 监听方法引用 |
| `FlowCondition` | `@start(and_("a", "b"))` | 复合条件 |

## 7.5 @listen() 装饰器：事件监听

`@listen()` 是 Flow 中使用最频繁的装饰器，它声明"当某个方法完成后，执行我"：

```python
# crewai/flow/flow.py

def listen(
    condition: str | FlowCondition | Callable[..., Any],
) -> Callable[[Callable[P, R]], ListenMethod[P, R]]:
    """Creates a listener that executes when specified conditions are met."""

    def decorator(func: Callable[P, R]) -> ListenMethod[P, R]:
        wrapper = ListenMethod(func)

        if is_flow_method_name(condition):
            wrapper.__trigger_methods__ = [condition]
            wrapper.__condition_type__ = OR_CONDITION
        elif is_flow_condition_dict(condition):
            if "conditions" in condition:
                wrapper.__trigger_condition__ = condition
                wrapper.__trigger_methods__ = _extract_all_methods(condition)
                wrapper.__condition_type__ = condition["type"]
            elif "methods" in condition:
                wrapper.__trigger_methods__ = condition["methods"]
                wrapper.__condition_type__ = condition["type"]
            else:
                raise ValueError(
                    "Condition dict must contain 'conditions' or 'methods'"
                )
        elif is_flow_method_callable(condition):
            wrapper.__trigger_methods__ = [condition.__name__]
            wrapper.__condition_type__ = OR_CONDITION
        else:
            raise ValueError(
                "Condition must be a method, string, or a result of or_() or and_()"
            )
        return wrapper

    return decorator
```

与 `@start()` 不同，`@listen()` 的 `condition` 参数是**必需的**——一个 listener 必须知道它在监听谁。

### 7.5.1 基本监听

最简单的用法是监听单个方法：

```python
class MyFlow(Flow):
    @start()
    def fetch_data(self):
        return {"items": [1, 2, 3]}

    @listen("fetch_data")
    def process_data(self, data):
        # data 是 fetch_data 的返回值
        return sum(data["items"])

    @listen("process_data")
    def report(self, total):
        print(f"Total: {total}")
```

关键点：listener 方法可以接受一个参数，该参数就是上游方法的返回值。执行引擎在调用 listener 之前会检查其签名：

```python
sig = inspect.signature(method)
params = list(sig.parameters.values())
method_params = [p for p in params if p.name != "self"]

if method_params:
    listener_result, finished_event_id = await self._execute_method(
        listener_name, method, result
    )
else:
    listener_result, finished_event_id = await self._execute_method(
        listener_name, method
    )
```

如果 listener 没有额外参数（除了 `self`），上游返回值就不会被传入。

### 7.5.2 监听方法引用

除了字符串名称，还可以直接传入方法引用：

```python
class MyFlow(Flow):
    @start()
    def step_one(self):
        return "data"

    @listen(step_one)  # 直接引用，而非 "step_one" 字符串
    def step_two(self, data):
        pass
```

源码中的判断逻辑为：

```python
elif is_flow_method_callable(condition):
    wrapper.__trigger_methods__ = [condition.__name__]
    wrapper.__condition_type__ = OR_CONDITION
```

方法引用最终还是被转换为方法名字符串。方法引用的好处是 IDE 可以提供自动补全和重构支持。

## 7.6 @router() 装饰器：条件路由

`@router()` 是 Flow 系统的"决策节点"。它监听上游方法，根据返回值决定下一步走哪条路径：

```python
# crewai/flow/flow.py

def router(
    condition: str | FlowCondition | Callable[..., Any],
) -> Callable[[Callable[P, R]], RouterMethod[P, R]]:
    """Creates a routing method that directs flow execution based on conditions."""

    def decorator(func: Callable[P, R]) -> RouterMethod[P, R]:
        wrapper = RouterMethod(func)

        if is_flow_method_name(condition):
            wrapper.__trigger_methods__ = [condition]
            wrapper.__condition_type__ = OR_CONDITION
        elif is_flow_condition_dict(condition):
            # ... 与 listen 相同的条件解析逻辑
        elif is_flow_method_callable(condition):
            wrapper.__trigger_methods__ = [condition.__name__]
            wrapper.__condition_type__ = OR_CONDITION
        else:
            raise ValueError(...)
        return wrapper

    return decorator
```

Router 的独特之处在于其返回值具有语义含义——它不仅是数据，更是路由指令。典型用法：

```python
class QAFlow(Flow):
    @start()
    def generate_answer(self):
        return self.state.answer

    @router("generate_answer")
    def check_quality(self, answer):
        if self.state.score > 0.8:
            return "PASS"
        return "RETRY"

    @listen("PASS")
    def publish(self):
        print("Publishing answer...")

    @listen("RETRY")
    def regenerate(self):
        print("Regenerating...")
```

当 `check_quality` 返回 `"PASS"` 时，执行引擎会寻找所有监听 `"PASS"` 的 listener 并触发它们。返回 `"RETRY"` 则走另一条路径。

### 7.6.1 Router 路径的静态分析

CrewAI 在类定义阶段就会尝试分析 router 方法可能返回的所有值，用于可视化和验证。这通过 AST 分析实现：

```python
# crewai/flow/utils.py（由 FlowMeta 调用）

def get_possible_return_constants(function: Any, verbose: bool = True) -> list[str] | None:
    """Extract possible string return values from a function using AST parsing."""
    unwrapped = _unwrap_function(function)
    source = inspect.getsource(function)
    source = textwrap.dedent(source)
    code_ast = ast.parse(source)

    return_values: set[str] = set()

    # 1. 检查返回类型注解：-> Literal["a", "b"] 或 -> MyEnum
    for node in ast.walk(code_ast):
        if isinstance(node, ast.FunctionDef):
            if node.returns:
                annotation_values = _extract_string_literals_from_type_annotation(
                    node.returns, function_globals
                )
                return_values.update(annotation_values)
            break

    # 2. 分析 return 语句中的字符串字面量
    # 3. 追踪变量赋值和字典查找
    # 4. 检查 self.state.attr 的比较操作
    ...
```

这意味着你可以用类型注解来声明 router 的路径：

```python
@router("validate")
def route(self) -> Literal["SUCCESS", "FAILURE"]:
    ...
```

AST 分析器会提取 `"SUCCESS"` 和 `"FAILURE"` 作为已知路径。

## 7.7 or_() 和 and_()：复合条件

实际业务中，一个方法可能需要在多个上游方法之一完成后触发（OR），或者在所有上游方法都完成后触发（AND）。CrewAI 提供了 `or_()` 和 `and_()` 两个条件构造函数。

### 7.7.1 OR 条件

```python
# crewai/flow/flow.py

def or_(*conditions: str | FlowCondition | Callable[..., Any]) -> FlowCondition:
    """Combines multiple conditions with OR logic."""
    processed_conditions: FlowConditions = []
    for condition in conditions:
        if is_flow_condition_dict(condition) or is_flow_method_name(condition):
            processed_conditions.append(condition)
        elif is_flow_method_callable(condition):
            processed_conditions.append(condition.__name__)
        else:
            raise ValueError("Invalid condition in or_()")
    return {"type": OR_CONDITION, "conditions": processed_conditions}
```

`or_()` 返回一个 `FlowCondition` 字典：`{"type": "OR", "conditions": [...]}`。使用示例：

```python
@listen(or_("fetch_from_api", "fetch_from_cache"))
def process(self, data):
    # fetch_from_api 或 fetch_from_cache 任一完成即触发
    pass
```

重要行为：多源 OR listener 只触发一次。当第一个上游完成时 listener 被触发并标记为已触发，后续上游完成时不再重复触发。这通过 `_fired_or_listeners` 集合跟踪：

```python
def _mark_or_listener_fired(self, listener_name: FlowMethodName) -> bool:
    with self._or_listeners_lock:
        if listener_name in self._fired_or_listeners:
            return False
        self._fired_or_listeners.add(listener_name)
        return True
```

### 7.7.2 AND 条件

```python
def and_(*conditions: str | FlowCondition | Callable[..., Any]) -> FlowCondition:
    """Combines multiple conditions with AND logic."""
    processed_conditions: FlowConditions = []
    for condition in conditions:
        if is_flow_condition_dict(condition) or is_flow_method_name(condition):
            processed_conditions.append(condition)
        elif is_flow_method_callable(condition):
            processed_conditions.append(condition.__name__)
        else:
            raise ValueError("Invalid condition in and_()")
    return {"type": AND_CONDITION, "conditions": processed_conditions}
```

AND 条件要求所有上游方法都完成后才触发。执行引擎通过 `_pending_and_listeners` 字典跟踪每个 AND listener 还在等待哪些方法：

```python
# _find_triggered_methods 中的 AND 处理逻辑
elif condition_type == AND_CONDITION:
    pending_key = PendingListenerKey(listener_name)
    if pending_key not in self._pending_and_listeners:
        self._pending_and_listeners[pending_key] = set(methods)
    if trigger_method in self._pending_and_listeners[pending_key]:
        self._pending_and_listeners[pending_key].discard(trigger_method)

    if not self._pending_and_listeners[pending_key]:
        triggered.append(listener_name)
        self._pending_and_listeners.pop(pending_key, None)
```

使用示例：

```python
@listen(and_("validate_format", "validate_content", "validate_permissions"))
def all_validations_passed(self):
    # 三个验证全部通过后才执行
    self.state.validated = True
```

### 7.7.3 嵌套条件

`or_()` 和 `and_()` 可以任意嵌套，形成复杂的触发条件树：

```python
@listen(or_(and_("step1", "step2"), "step3"))
def complex_handler(self):
    # (step1 AND step2) OR step3
    pass

@listen(and_(or_("api_result", "cache_result"), "auth_check"))
def another_handler(self):
    # (api_result OR cache_result) AND auth_check
    pass
```

嵌套条件的评估通过递归实现：

```python
def _evaluate_condition(
    self,
    condition: FlowMethodName | FlowCondition,
    trigger_method: FlowMethodName,
    listener_name: FlowMethodName,
) -> bool:
    if is_flow_method_name(condition):
        return condition == trigger_method

    if is_flow_condition_dict(condition):
        normalized = _normalize_condition(condition)
        cond_type = normalized.get("type", OR_CONDITION)
        sub_conditions = normalized.get("conditions", [])

        if cond_type == OR_CONDITION:
            return any(
                self._evaluate_condition(sub_cond, trigger_method, listener_name)
                for sub_cond in sub_conditions
            )

        if cond_type == AND_CONDITION:
            pending_key = PendingListenerKey(f"{listener_name}:{id(condition)}")
            # ... 跟踪待完成方法
```

每个嵌套的 AND 条件都有独立的 `pending_key`（格式为 `"listener_name:object_id"`），避免不同层级的 AND 条件相互干扰。

## 7.8 FlowMeta 元类：编译时的图构建

装饰器只是在方法上附加了元数据，真正将这些元数据"编译"成触发关系图的是 `FlowMeta` 元类：

```python
# crewai/flow/flow.py

class FlowMeta(type):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        cls = super().__new__(mcs, name, bases, namespace)

        start_methods = []
        listeners = {}
        router_paths = {}
        routers = set()

        for attr_name, attr_value in namespace.items():
            if (
                hasattr(attr_value, "__is_flow_method__")
                or hasattr(attr_value, "__is_start_method__")
                or hasattr(attr_value, "__trigger_methods__")
                or hasattr(attr_value, "__is_router__")
            ):
                # 注册 start 方法
                if hasattr(attr_value, "__is_start_method__"):
                    start_methods.append(attr_name)

                # 注册 listeners 和 routers
                if (
                    hasattr(attr_value, "__trigger_methods__")
                    and attr_value.__trigger_methods__ is not None
                ):
                    methods = attr_value.__trigger_methods__
                    condition_type = getattr(
                        attr_value, "__condition_type__", OR_CONDITION
                    )

                    if (
                        hasattr(attr_value, "__trigger_condition__")
                        and attr_value.__trigger_condition__ is not None
                    ):
                        listeners[attr_name] = attr_value.__trigger_condition__
                    else:
                        listeners[attr_name] = (condition_type, methods)

                    if hasattr(attr_value, "__is_router__") and attr_value.__is_router__:
                        routers.add(attr_name)
                        possible_returns = get_possible_return_constants(attr_value)
                        if possible_returns:
                            router_paths[attr_name] = possible_returns
                        else:
                            router_paths[attr_name] = []

        cls._start_methods = start_methods
        cls._listeners = listeners
        cls._routers = routers
        cls._router_paths = router_paths

        return cls
```

`FlowMeta.__new__` 在类**定义**时执行（不是实例化时），它遍历类的命名空间，根据方法上的标志位将它们分类到四个类级别属性中：

| 属性 | 类型 | 含义 |
|------|------|------|
| `_start_methods` | `list[str]` | 所有入口方法名 |
| `_listeners` | `dict[str, condition]` | 方法名 -> 触发条件的映射 |
| `_routers` | `set[str]` | 所有 router 方法名 |
| `_router_paths` | `dict[str, list[str]]` | router -> 可能返回值的映射 |

`_listeners` 的值有两种形式：
- **简单条件**：`(condition_type, methods)` 元组，如 `("OR", ["method_a"])`
- **复合条件**：`FlowCondition` 字典，如 `{"type": "AND", "conditions": ["a", "b"]}`

这些类级别属性对所有实例共享，在运行时被执行引擎读取以构建执行图。

## 7.9 Flow 类的初始化

当实例化一个 Flow 时，`__init__` 方法完成以下关键初始化：

```python
class Flow(Generic[T], metaclass=FlowMeta):
    def __init__(
        self,
        persistence: FlowPersistence | None = None,
        tracing: bool | None = None,
        suppress_flow_events: bool = False,
        max_method_calls: int = 100,
        **kwargs: Any,
    ) -> None:
        self._methods: dict[FlowMethodName, FlowMethod[Any, Any]] = {}
        self._method_execution_counts: dict[FlowMethodName, int] = {}
        self._pending_and_listeners: dict[PendingListenerKey, set[FlowMethodName]] = {}
        self._fired_or_listeners: set[FlowMethodName] = set()
        self._method_outputs: list[Any] = []
        self._state_lock = threading.Lock()
        self._completed_methods: set[FlowMethodName] = set()
        self._method_call_counts: dict[FlowMethodName, int] = {}
        self._max_method_calls = max_method_calls

        self._state = self._create_initial_state()

        # 注册所有 flow 方法
        for method_name in dir(self):
            if not method_name.startswith("_"):
                method = getattr(self, method_name)
                if is_flow_method(method):
                    if not hasattr(method, "__self__"):
                        method = method.__get__(self, self.__class__)
                    self._methods[method.__name__] = method
```

注意 `_max_method_calls` 参数（默认 100），它是防止无限循环的安全阀——如果同一个方法在一次 Flow 执行中被调用超过 100 次，会抛出 `RecursionError`。

## 7.10 类型定义体系

`flow/types.py` 定义了 Flow 系统的类型基础设施：

```python
# crewai/flow/types.py

FlowMethodName = NewType("FlowMethodName", str)
FlowRouteName = NewType("FlowRouteName", str)
PendingListenerKey = NewType(
    "PendingListenerKey",
    Annotated[str, "nested flow conditions use 'listener_name:object_id'"],
)
```

`FlowMethodName` 是 `str` 的 NewType 别名，在运行时等价于 `str`，但在静态类型检查时提供了更强的类型约束。`PendingListenerKey` 的注解说明了其特殊格式——对于嵌套条件，key 的格式为 `"listener_name:object_id"`，以区分同一 listener 的不同嵌套 AND 条件。

`types.py` 还定义了执行数据的结构，用于持久化和恢复：

```python
class FlowExecutionData(TypedDict):
    id: str
    flow: FlowData
    inputs: dict[str, Any]
    completed_methods: list[CompletedMethodData]
    execution_methods: list[ExecutionMethodData]
```

这些 TypedDict 定义了 Flow 执行状态的完整快照格式，支持序列化到 JSON 并在之后恢复执行。

## 7.11 常量与工具函数

`flow/constants.py` 只定义了两个常量：

```python
AND_CONDITION: Final[Literal["AND"]] = "AND"
OR_CONDITION: Final[Literal["OR"]] = "OR"
```

使用 `Final[Literal[...]]` 确保它们在运行时不可修改，在类型检查时被约束为字面量类型。

`flow/utils.py` 中的辅助函数承担了条件解析和图分析的职责。其中 `_normalize_condition` 将各种格式的条件统一为标准形式：

```python
def _normalize_condition(
    condition: FlowConditions | FlowCondition | FlowMethodName,
) -> FlowCondition:
    if is_flow_method_name(condition):
        return {"type": OR_CONDITION, "conditions": [condition]}
    if is_flow_condition_dict(condition):
        if "conditions" in condition:
            return condition
        if "methods" in condition:
            return {"type": condition["type"], "conditions": condition["methods"]}
        return condition
    if is_flow_condition_list(condition):
        return {"type": OR_CONDITION, "conditions": condition}
    raise ValueError(f"Cannot normalize condition: {condition}")
```

这个函数确保无论输入是字符串、字典还是列表，输出都是统一的 `{"type": ..., "conditions": [...]}` 格式，简化了下游代码的处理逻辑。

## 7.12 上下文变量

`flow/flow_context.py` 使用 Python 的 `contextvars` 模块管理运行时上下文：

```python
# crewai/flow/flow_context.py

current_flow_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "flow_request_id", default=None
)

current_flow_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "flow_id", default=None
)

current_flow_method_name: contextvars.ContextVar[str] = contextvars.ContextVar(
    "flow_method_name", default="unknown"
)
```

`ContextVar` 是 Python 对异步任务友好的线程局部存储。在 Flow 执行期间：
- `current_flow_id` 标识当前执行的 Flow 实例
- `current_flow_request_id` 标识当前请求（可能跨多个 Flow）
- `current_flow_method_name` 标识当前正在执行的方法

这些上下文变量在方法执行前设置、执行后重置，确保嵌套 Flow 和并发执行不会相互干扰。

## 本章要点

- **FlowState** 继承自 Pydantic `BaseModel`，自动生成 UUID，提供类型验证和序列化能力；Flow 也支持普通 `dict` 作为 state。
- **StateProxy** 通过描述符协议透明代理 state 访问，对 `list`、`dict` 类型字段自动包装为线程安全的 `LockedListProxy` / `LockedDictProxy`。
- **FlowMethod** 是所有装饰器的包装基类，通过 `functools.update_wrapper` 保留原方法信息，通过 `__get__` 实现描述符协议支持实例绑定。
- **@start()** 标记入口方法，支持无条件启动和条件启动两种模式；`kickoff()` 时所有无条件 start 方法被并行执行。
- **@listen(condition)** 声明事件监听关系，被监听方法的返回值会自动传递给 listener（如果其签名接受参数）。
- **@router(condition)** 是特殊的 listener，其返回值作为路由标签触发下游的 `@listen("LABEL")` 方法；CrewAI 通过 AST 分析静态提取 router 可能的返回值。
- **or_() / and_()** 构造复合条件，支持任意嵌套；OR 语义为"任一满足即触发"（且多源 OR 只触发一次），AND 语义为"全部满足才触发"。
- **FlowMeta 元类** 在类定义时遍历命名空间，将装饰器元数据编译为 `_start_methods`、`_listeners`、`_routers`、`_router_paths` 四个类级别属性，构建触发关系图。
- **contextvars** 提供异步安全的运行时上下文，跟踪当前 flow ID、request ID 和 method name。
- **max_method_calls** 参数（默认 100）作为安全阀防止循环 Flow 中的无限递归。
