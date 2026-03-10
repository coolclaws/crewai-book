# 第 9 章　Flow 持久化

在前面几章中，我们深入分析了 Flow 的执行引擎、状态管理和事件驱动机制。然而，所有讨论都建立在一个前提之上：Flow 的执行是一次性的，状态仅存在于内存中。一旦进程退出，所有中间状态都会丢失。

对于生产环境中的长时间运行工作流，这是不可接受的。想象一个需要人类审批的流程：工作流执行到某个节点后暂停，等待人类通过外部接口提供反馈，然后从断点恢复执行。这要求系统能够将 Flow 的状态持久化到磁盘，并在需要时重新加载。

CrewAI 通过 `flow/persistence/` 模块提供了一套完整的持久化方案。本章将从抽象接口到具体实现，从 decorator 到恢复机制，全面剖析这一子系统。

## 9.1 模块概览

持久化模块位于 `crewai/flow/persistence/` 目录下，包含以下文件：

| 文件 | 职责 |
|------|------|
| `__init__.py` | 导出公共 API |
| `base.py` | 定义 `FlowPersistence` 抽象基类 |
| `decorators.py` | 实现 `@persist` decorator |
| `sqlite.py` | 提供 `SQLiteFlowPersistence` 实现 |

模块的公共 API 非常简洁：

```python
from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.persistence.decorators import persist
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence

__all__ = ["FlowPersistence", "SQLiteFlowPersistence", "persist"]

StateType = TypeVar("StateType", bound=dict[str, Any] | BaseModel)
DictStateType = dict[str, Any]
```

三个导出名称对应持久化子系统的三个核心概念：抽象接口、默认实现、以及将两者串联起来的 decorator。

## 9.2 FlowPersistence 抽象基类

`FlowPersistence` 是持久化子系统的核心契约。所有持久化后端都必须遵循这个接口，无论底层存储是 SQLite、PostgreSQL、Redis 还是云端对象存储。

### 9.2.1 接口定义

```python
class FlowPersistence(ABC):
    """Abstract base class for flow state persistence.

    This class defines the interface that all persistence implementations
    must follow. It supports both structured (Pydantic BaseModel) and
    unstructured (dict) states.
    """

    @abstractmethod
    def init_db(self) -> None:
        """Initialize the persistence backend."""

    @abstractmethod
    def save_state(
        self,
        flow_uuid: str,
        method_name: str,
        state_data: dict[str, Any] | BaseModel,
    ) -> None:
        """Persist the flow state after method completion."""

    @abstractmethod
    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        """Load the most recent state for a given flow UUID."""
```

三个抽象方法构成了最小持久化协议：

1. **`init_db()`**：初始化后端存储。对于 SQLite，这意味着创建表和索引；对于 Redis，可能是测试连接和设置 key prefix。
2. **`save_state(flow_uuid, method_name, state_data)`**：将状态数据持久化。`flow_uuid` 是 Flow 实例的唯一标识，`method_name` 记录触发保存的方法名，`state_data` 是实际的状态数据。
3. **`load_state(flow_uuid)`**：根据 `flow_uuid` 加载最近一次保存的状态。返回 `None` 表示没有找到历史状态。

注意 `state_data` 参数接受 `dict[str, Any] | BaseModel` 联合类型。这与 Flow 的双模式状态设计一致——用户可以使用无类型的 dict 状态，也可以使用 Pydantic BaseModel 定义的结构化状态。持久化层需要兼容两种模式。

### 9.2.2 异步人类反馈支持

除了三个核心抽象方法，`FlowPersistence` 还定义了三个可选方法，用于支持异步人类反馈（async Human-in-the-Loop）场景：

```python
def save_pending_feedback(
    self,
    flow_uuid: str,
    context: PendingFeedbackContext,
    state_data: dict[str, Any] | BaseModel,
) -> None:
    """Save state with a pending feedback marker."""
    # Default: just save the state without pending context
    self.save_state(flow_uuid, context.method_name, state_data)

def load_pending_feedback(
    self,
    flow_uuid: str,
) -> tuple[dict[str, Any], PendingFeedbackContext] | None:
    """Load state and pending feedback context."""
    return None

def clear_pending_feedback(self, flow_uuid: str) -> None:
    """Clear the pending feedback marker after successful resume."""
```

这三个方法不是抽象的——它们提供了默认实现。基类的 `save_pending_feedback()` 默认退化为普通的 `save_state()`，`load_pending_feedback()` 默认返回 `None`，`clear_pending_feedback()` 默认什么都不做。

这种设计体现了接口隔离原则（Interface Segregation Principle）。如果你的应用不需要异步人类反馈功能，只需实现三个核心抽象方法即可。只有在需要 HITL（Human-in-the-Loop）支持时，才需要 override 这三个可选方法。

### 9.2.3 设计分析

`FlowPersistence` 的设计有几个值得注意的特点：

**以 `flow_uuid` 为索引键。** 每个 Flow 实例通过状态中的 `id` 字段唯一标识。这个 ID 是持久化查找的唯一键。一个 Flow 执行过程中可能保存多次状态（每个方法执行后保存一次），但恢复时只加载最新的一条。

**方法名作为元数据。** `save_state()` 接收 `method_name` 参数，但 `load_state()` 不需要它。方法名被记录下来主要用于审计和调试——你可以查看某个 Flow 实例的完整执行轨迹，知道哪些方法在什么时候被执行了。

**单向时间线。** `load_state()` 总是返回最新状态，没有提供按时间点查询的能力。这是一个有意识的简化——持久化的目的是断点恢复，而不是状态回溯。

## 9.3 SQLiteFlowPersistence 实现

SQLite 后端是 CrewAI 提供的默认（也是目前唯一的内置）持久化实现。选择 SQLite 是合理的：它是 Python 标准库的一部分，不需要额外部署数据库服务，对于开发、测试和中等规模的生产场景都足够用。

### 9.3.1 初始化与数据库 Schema

```python
class SQLiteFlowPersistence(FlowPersistence):

    def __init__(self, db_path: str | None = None) -> None:
        path = db_path or str(Path(db_storage_path()) / "flow_states.db")
        if not path:
            raise ValueError("Database path must be provided")
        self.db_path = path
        self.init_db()

    def init_db(self) -> None:
        """Create the necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS flow_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    flow_uuid TEXT NOT NULL,
                    method_name TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    state_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_flow_states_uuid
                ON flow_states(flow_uuid)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    flow_uuid TEXT NOT NULL UNIQUE,
                    context_json TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at DATETIME NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pending_feedback_uuid
                ON pending_feedback(flow_uuid)
            """)
```

数据库包含两张表：

**`flow_states` 表** 是主状态表，采用追加写入（append-only）模式。每次 `save_state()` 调用都会 INSERT 一条新记录，而不是 UPDATE 已有记录。这意味着一个 Flow 实例的完整执行历史都被保留了下来。`flow_uuid` 上建有索引以加速查询。

**`pending_feedback` 表** 专门用于异步人类反馈场景。注意 `flow_uuid` 列有 `UNIQUE` 约束——每个 Flow 实例在任一时刻最多只能有一个待处理的反馈请求。这个约束通过后续的 `INSERT OR REPLACE` 语义来维护。

构造函数的行为值得关注：如果不传入 `db_path`，它会使用 `db_storage_path()` 工具函数获取默认存储路径，并在其中创建 `flow_states.db` 文件。`init_db()` 在构造函数中直接调用，确保实例创建时数据库已经就绪。

### 9.3.2 状态保存

```python
def save_state(
    self,
    flow_uuid: str,
    method_name: str,
    state_data: dict[str, Any] | BaseModel,
) -> None:
    if isinstance(state_data, BaseModel):
        state_dict = state_data.model_dump()
    elif isinstance(state_data, dict):
        state_dict = state_data
    else:
        raise ValueError(
            f"state_data must be either a Pydantic BaseModel or dict, "
            f"got {type(state_data)}"
        )

    with sqlite3.connect(self.db_path) as conn:
        conn.execute("""
            INSERT INTO flow_states (
                flow_uuid, method_name, timestamp, state_json
            ) VALUES (?, ?, ?, ?)
        """, (
            flow_uuid,
            method_name,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(state_dict),
        ))
```

保存逻辑清晰明了：

1. 首先将 `state_data` 统一转换为 dict。如果是 Pydantic BaseModel，调用 `model_dump()` 序列化；如果已经是 dict，直接使用。
2. 将 dict 通过 `json.dumps()` 序列化为 JSON 字符串存入数据库。
3. 时间戳使用 UTC 时区的 ISO 格式字符串。

这里有一个潜在的限制：`json.dumps()` 对于包含自定义对象的状态数据可能会失败。如果你的 Flow 状态中包含无法 JSON 序列化的对象（比如 datetime 或自定义类实例），需要在 Pydantic model 中定义合适的序列化器。

### 9.3.3 状态加载

```python
def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.execute("""
            SELECT state_json
            FROM flow_states
            WHERE flow_uuid = ?
            ORDER BY id DESC
            LIMIT 1
        """, (flow_uuid,))
        row = cursor.fetchone()

    if row:
        result = json.loads(row[0])
        return result if isinstance(result, dict) else None
    return None
```

加载逻辑利用 `ORDER BY id DESC LIMIT 1` 获取最新的一条记录。由于 `id` 是自增主键，按 `id` 降序排列等价于按时间降序排列。这比按 `timestamp` 排序更可靠（避免了时钟回拨等问题）也更高效（主键索引天然有序）。

返回值经过一次类型检查：只有当 `json.loads()` 的结果确实是 dict 时才返回，否则返回 `None`。这是一种防御性编程——防止数据库中存储了格式错误的 JSON。

### 9.3.4 异步反馈持久化

```python
def save_pending_feedback(
    self,
    flow_uuid: str,
    context: PendingFeedbackContext,
    state_data: dict[str, Any] | BaseModel,
) -> None:
    if isinstance(state_data, BaseModel):
        state_dict = state_data.model_dump()
    elif isinstance(state_data, dict):
        state_dict = state_data
    else:
        raise ValueError(...)

    # Also save to regular state table for consistency
    self.save_state(flow_uuid, context.method_name, state_data)

    # Save pending feedback context
    with sqlite3.connect(self.db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO pending_feedback (
                flow_uuid, context_json, state_json, created_at
            ) VALUES (?, ?, ?, ?)
        """, (
            flow_uuid,
            json.dumps(context.to_dict()),
            json.dumps(state_dict),
            datetime.now(timezone.utc).isoformat(),
        ))
```

`save_pending_feedback()` 做了两件事：

1. 先调用 `self.save_state()` 将状态保存到主表，确保一致性。
2. 再将反馈上下文和状态一起保存到 `pending_feedback` 表。`INSERT OR REPLACE` 确保同一个 `flow_uuid` 只有一条待处理记录——如果之前有旧的 pending feedback，会被新的替换掉。

对应的加载和清除方法同样直观：

```python
def load_pending_feedback(
    self, flow_uuid: str,
) -> tuple[dict[str, Any], PendingFeedbackContext] | None:
    from crewai.flow.async_feedback.types import PendingFeedbackContext

    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.execute("""
            SELECT state_json, context_json
            FROM pending_feedback
            WHERE flow_uuid = ?
        """, (flow_uuid,))
        row = cursor.fetchone()

    if row:
        state_dict = json.loads(row[0])
        context_dict = json.loads(row[1])
        context = PendingFeedbackContext.from_dict(context_dict)
        return (state_dict, context)
    return None

def clear_pending_feedback(self, flow_uuid: str) -> None:
    with sqlite3.connect(self.db_path) as conn:
        conn.execute("""
            DELETE FROM pending_feedback WHERE flow_uuid = ?
        """, (flow_uuid,))
```

注意 `PendingFeedbackContext` 的 import 使用了延迟导入（在方法内部 import），这是为了避免循环依赖。

## 9.4 @persist Decorator

`@persist` decorator 是持久化子系统的使用入口。它将持久化逻辑透明地注入到 Flow 方法中，让开发者不需要手动调用 `save_state()`。

### 9.4.1 双层装饰模式

`@persist` 的设计支持两种使用方式：**类级别装饰**和**方法级别装饰**。

```python
# 方法级别：只持久化特定方法
class MyFlow(Flow):
    @start()
    @persist(SQLiteFlowPersistence())
    def sync_method(self):
        pass

# 类级别：自动持久化所有 Flow 方法
@persist(verbose=True)
class MyFlow(Flow[MyState]):
    @start()
    def begin(self):
        pass
```

这种灵活性通过在 decorator 内部检查装饰目标的类型来实现：

```python
def persist(
    persistence: FlowPersistence | None = None,
    verbose: bool = False,
) -> Callable[[type | Callable[..., T]], type | Callable[..., T]]:

    def decorator(target: type | Callable[..., T]) -> type | Callable[..., T]:
        actual_persistence = persistence or SQLiteFlowPersistence()

        if isinstance(target, type):
            # Class decoration path
            ...
        else:
            # Method decoration path
            ...

    return decorator
```

`isinstance(target, type)` 是关键判断——如果 `target` 是一个类，走类装饰路径；否则走方法装饰路径。

### 9.4.2 类级别装饰

类级别装饰的逻辑较为复杂，因为它需要自动为所有 Flow 方法注入持久化逻辑：

```python
if isinstance(target, type):
    original_init = target.__init__

    @functools.wraps(original_init)
    def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
        if "persistence" not in kwargs:
            kwargs["persistence"] = actual_persistence
        original_init(self, *args, **kwargs)

    target.__init__ = new_init

    # Find all flow methods
    original_methods = {
        name: method
        for name, method in target.__dict__.items()
        if callable(method)
        and (
            hasattr(method, "__is_start_method__")
            or hasattr(method, "__trigger_methods__")
            or hasattr(method, "__condition_type__")
            or hasattr(method, "__is_flow_method__")
            or hasattr(method, "__is_router__")
        )
    }

    # Wrap each method with persistence
    for name, method in original_methods.items():
        if asyncio.iscoroutinefunction(method):
            # Async wrapper
            ...
        else:
            # Sync wrapper
            ...
```

这里有三个关键步骤：

**第一步：修改 `__init__`。** 通过替换类的 `__init__` 方法，在实例化时自动注入 `persistence` 参数。如果用户手动传入了 `persistence`，就使用用户的；否则使用 decorator 指定的默认值。

**第二步：识别 Flow 方法。** 通过检查方法上的特殊属性（`__is_start_method__`、`__trigger_methods__`、`__condition_type__` 等）来识别哪些方法是 Flow 方法。只有这些方法才需要注入持久化逻辑。

**第三步：创建 wrapper。** 对于每个识别出的 Flow 方法，创建一个 wrapper 函数，在原方法执行后调用持久化逻辑。同步方法和异步方法分别处理。

Wrapper 的创建使用了闭包工厂模式，以避免 Python 循环变量捕获的经典陷阱：

```python
def create_sync_wrapper(
    method_name: str, original_method: Callable[..., Any]
) -> Callable[..., Any]:
    @functools.wraps(original_method)
    def method_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_method(self, *args, **kwargs)
        PersistenceDecorator.persist_state(
            self, method_name, actual_persistence, verbose
        )
        return result
    return method_wrapper
```

特别重要的是 wrapper 保留了原方法的所有 decorator 属性：

```python
for attr in [
    "__is_start_method__",
    "__trigger_methods__",
    "__condition_type__",
    "__is_router__",
]:
    if hasattr(method, attr):
        setattr(wrapped, attr, getattr(method, attr))
wrapped.__is_flow_method__ = True
```

这确保了 Flow 引擎在运行时仍然能够识别这些方法的角色（start、listen、router 等）。如果属性丢失，Flow 的事件驱动调度将无法正常工作。

### 9.4.3 方法级别装饰

方法级别装饰更简单，因为只需要处理单个方法：

```python
# Method decoration
method = target
method.__is_flow_method__ = True

if asyncio.iscoroutinefunction(method):
    @functools.wraps(method)
    async def method_async_wrapper(
        flow_instance: Any, *args: Any, **kwargs: Any
    ) -> T:
        method_coro = method(flow_instance, *args, **kwargs)
        if asyncio.iscoroutine(method_coro):
            result = await method_coro
        else:
            result = method_coro
        PersistenceDecorator.persist_state(
            flow_instance, method.__name__, actual_persistence, verbose
        )
        return cast(T, result)
    # ... preserve attributes ...
    return cast(Callable[..., T], method_async_wrapper)

@functools.wraps(method)
def method_sync_wrapper(flow_instance: Any, *args: Any, **kwargs: Any) -> T:
    result = method(flow_instance, *args, **kwargs)
    PersistenceDecorator.persist_state(
        flow_instance, method.__name__, actual_persistence, verbose
    )
    return result
```

注意异步 wrapper 中有一个防御性检查：`if asyncio.iscoroutine(method_coro)`。即使 `iscoroutinefunction()` 判断方法是异步的，它仍然检查调用结果是否确实是 coroutine。这种双重检查增加了鲁棒性。

### 9.4.4 PersistenceDecorator 辅助类

实际的持久化逻辑被封装在 `PersistenceDecorator` 类中：

```python
class PersistenceDecorator:
    _printer: ClassVar[Printer] = Printer()

    @classmethod
    def persist_state(
        cls,
        flow_instance: Flow[Any],
        method_name: str,
        persistence_instance: FlowPersistence,
        verbose: bool = False,
    ) -> None:
        try:
            state = getattr(flow_instance, "state", None)
            if state is None:
                raise ValueError("Flow instance has no state")

            flow_uuid: str | None = None
            if isinstance(state, dict):
                flow_uuid = state.get("id")
            elif hasattr(state, "_unwrap"):
                unwrapped = state._unwrap()
                if isinstance(unwrapped, dict):
                    flow_uuid = unwrapped.get("id")
                else:
                    flow_uuid = getattr(unwrapped, "id", None)
            elif isinstance(state, BaseModel) or hasattr(state, "id"):
                flow_uuid = getattr(state, "id", None)

            if not flow_uuid:
                raise ValueError(
                    "Flow state must have an 'id' field for persistence"
                )

            try:
                state_data = (
                    state._unwrap() if hasattr(state, "_unwrap") else state
                )
                persistence_instance.save_state(
                    flow_uuid=flow_uuid,
                    method_name=method_name,
                    state_data=state_data,
                )
            except Exception as e:
                raise RuntimeError(
                    f"State persistence failed: {e!s}"
                ) from e
        except AttributeError as e:
            raise ValueError("Flow instance has no state") from e
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Flow state must have an 'id' field for persistence"
            ) from e
```

这个方法揭示了持久化的一个关键要求：**Flow 状态必须包含 `id` 字段**。没有 `id` 的状态无法被持久化，因为系统需要一个唯一标识来索引和检索状态。

提取 `id` 的逻辑处理了三种情况：
1. 状态是 dict，直接通过 `state.get("id")` 获取
2. 状态被包装过（有 `_unwrap()` 方法），先解包再获取
3. 状态是 Pydantic BaseModel 或有 `id` 属性的对象，通过 `getattr` 获取

## 9.5 断点恢复机制

持久化的最终目的是实现断点恢复。Flow 类提供了 `from_pending()` 类方法来完成这一功能：

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

    # Load pending feedback context and state
    loaded = persistence.load_pending_feedback(flow_id)
    if loaded is None:
        raise ValueError(
            f"No pending feedback found for flow_id: {flow_id}"
        )

    state_data, pending_context = loaded

    # Create flow instance with persistence
    instance = cls(persistence=persistence, **kwargs)

    # Restore state
    instance._initialize_state(state_data)

    # Store pending context for resume
    instance._pending_feedback_context = pending_context

    # Mark that we're resuming execution
    instance._is_execution_resuming = True

    # Mark the method as completed (it ran before pausing)
    instance._completed_methods.add(
        FlowMethodName(pending_context.method_name)
    )

    return instance
```

恢复流程清晰地分为五步：

1. **加载持久化数据**：通过 `load_pending_feedback()` 同时获取状态数据和待处理的反馈上下文
2. **创建新实例**：调用 `cls()` 创建一个新的 Flow 实例，并将 persistence 传入
3. **恢复状态**：通过 `_initialize_state()` 用持久化的数据初始化状态
4. **设置恢复标记**：将 `_is_execution_resuming` 设为 `True`，通知执行引擎这是一次恢复而非全新执行
5. **标记已完成方法**：将暂停前已执行的方法添加到 `_completed_methods` 集合，防止重复执行

典型的使用模式如下：

```python
# 第一阶段：启动 Flow，在需要人类反馈时暂停
try:
    flow = MyFlow(persistence=SQLiteFlowPersistence("flows.db"))
    result = flow.kickoff()
except HumanFeedbackPending as e:
    print(f"Flow paused, waiting for feedback: {e.context.flow_id}")

# 第二阶段：稍后恢复执行（可能是不同的进程）
flow = MyFlow.from_pending("abc-123", SQLiteFlowPersistence("flows.db"))
result = flow.resume("looks good!")
```

这种模式使得 Flow 可以跨进程、甚至跨机器恢复执行。只要新进程能访问同一个 SQLite 数据库文件（或使用同一个持久化后端），就能无缝恢复。

## 9.6 Flow 构造函数中的持久化集成

回顾 Flow 类的构造函数，可以看到持久化是一个一等公民（first-class citizen）：

```python
class Flow(Generic[T]):
    def __init__(
        self,
        persistence: FlowPersistence | None = None,
        tracing: bool | None = None,
        suppress_flow_events: bool = False,
        max_method_calls: int = 100,
    ) -> None:
        ...
        self._persistence: FlowPersistence | None = persistence
        self._is_execution_resuming: bool = False
        ...
```

`persistence` 参数是可选的——不传入时 Flow 不会进行任何持久化。这是一个好的默认值：对于不需要持久化的简单工作流，零配置即可运行；需要持久化时，显式传入后端实例。

`_is_execution_resuming` 标志告诉执行引擎当前是恢复模式还是正常模式，这影响了事件调度的行为——恢复模式下，已完成的方法不会被重新触发。

## 9.7 自定义持久化后端

得益于 `FlowPersistence` 的抽象设计，实现自定义后端非常简单。以下是一个基于 Redis 的持久化后端示例（伪代码）：

```python
import json
import redis
from crewai.flow.persistence.base import FlowPersistence

class RedisFlowPersistence(FlowPersistence):
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.client = redis.from_url(redis_url)
        self.init_db()

    def init_db(self) -> None:
        # Redis 不需要创建 schema，可以做连通性检查
        self.client.ping()

    def save_state(self, flow_uuid, method_name, state_data):
        if isinstance(state_data, BaseModel):
            state_dict = state_data.model_dump()
        else:
            state_dict = state_data

        record = {
            "method_name": method_name,
            "state": state_dict,
        }
        # 使用 LPUSH 保持追加写入语义
        self.client.lpush(
            f"flow:{flow_uuid}:states",
            json.dumps(record),
        )

    def load_state(self, flow_uuid):
        # LINDEX 0 获取最新记录
        data = self.client.lindex(f"flow:{flow_uuid}:states", 0)
        if data:
            record = json.loads(data)
            return record["state"]
        return None
```

只需要实现三个核心方法，就可以将 Flow 的状态存储到任何后端。

## 9.8 持久化数据流全景

将所有组件串联起来，持久化的完整数据流如下：

```
Flow 方法执行完毕
    ↓
@persist wrapper 拦截返回值
    ↓
PersistenceDecorator.persist_state()
    ↓
从 flow.state 提取 id
    ↓
调用 persistence.save_state(flow_uuid, method_name, state_data)
    ↓
SQLiteFlowPersistence.save_state()
    ↓
state → model_dump() / dict → json.dumps() → INSERT INTO flow_states
    ↓
SQLite 数据库文件

--- 恢复时 ---

MyFlow.from_pending(flow_id, persistence)
    ↓
persistence.load_pending_feedback(flow_id)
    ↓
SELECT state_json, context_json FROM pending_feedback
    ↓
json.loads() → (state_dict, PendingFeedbackContext)
    ↓
cls(persistence=persistence)
    ↓
instance._initialize_state(state_data)
    ↓
Flow 实例恢复完毕，可调用 resume()
```

## 本章要点

- **`FlowPersistence`** 是持久化的抽象基类，定义了 `init_db()`、`save_state()` 和 `load_state()` 三个核心抽象方法，以及三个可选的异步反馈方法
- **`SQLiteFlowPersistence`** 是默认的 SQLite 实现，使用两张表：`flow_states`（追加写入的状态历史）和 `pending_feedback`（异步人类反馈上下文）
- **`@persist` decorator** 支持类级别和方法级别两种装饰模式。类级别装饰自动为所有 Flow 方法注入持久化；方法级别装饰只影响单个方法
- **持久化的关键要求**：Flow 状态必须包含 `id` 字段，作为持久化存储和检索的唯一标识
- **断点恢复** 通过 `Flow.from_pending()` 类方法实现，它从持久化后端加载状态和反馈上下文，创建一个可以调用 `resume()` 的 Flow 实例
- **`PersistenceDecorator`** 辅助类封装了状态提取和保存的逻辑，包括处理 dict、Pydantic BaseModel 和包装状态三种情况
- 持久化层的设计遵循**接口隔离原则**：核心功能只需三个方法，HITL 支持是可选扩展
- 追加写入（append-only）的存储模式保留了完整的执行历史，但加载时只取最新状态
- 自定义持久化后端只需继承 `FlowPersistence` 并实现三个抽象方法即可
