# 第 22 章　Project 模板与 @CrewBase

前一章我们从 CLI 的角度看到了 CrewAI 项目的创建和运行。本章深入 `project/` 模块的源码，剖析 `@CrewBase` 装饰器如何将一个普通的 Python 类转变为声明式的 Crew 定义，以及 YAML 配置文件如何与 Python 代码协同工作。

## 22.1 project/ 模块概览

`project/` 目录包含四个核心文件：

| 文件 | 职责 |
|------|------|
| `crew_base.py` | `@CrewBase` 装饰器和 `CrewBaseMeta` 元类 |
| `annotations.py` | `@agent`、`@task`、`@crew` 等装饰器 |
| `wrappers.py` | 装饰器的 Wrapper 类型定义 |
| `utils.py` | `memoize` 缓存工具 |

这四个文件共同构成了 CrewAI 的"声明式项目"系统——开发者通过装饰器和 YAML 配置来定义 Crew，而非手动组装对象。

## 22.2 @CrewBase 装饰器：从类到 Crew

`@CrewBase` 是整个项目系统的核心。让我们先看它的使用方式：

```python
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class MyCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(config=self.agents_config['researcher'])

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config['research_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks)
```

### 22.2.1 元类机制

`@CrewBase` 的实现分为三层：

**第一层：`_CrewBaseType` 元类**

`CrewBase` 类本身使用 `_CrewBaseType` 作为元类，使得 `CrewBase` 可以作为装饰器使用：

```python
class _CrewBaseType(type):
    """Metaclass for CrewBase that makes it callable as a decorator."""

    def __call__(cls, decorated_cls: type) -> type[CrewClass]:
        __name = str(decorated_cls.__name__)
        __bases = tuple(decorated_cls.__bases__)
        __dict = {
            key: value
            for key, value in decorated_cls.__dict__.items()
            if key not in ("__dict__", "__weakref__")
        }
        __dict["__metaclass__"] = CrewBaseMeta
        return cast(type[CrewClass], CrewBaseMeta(__name, __bases, __dict))

class CrewBase(metaclass=_CrewBaseType):
    """Class decorator that applies CrewBaseMeta metaclass."""
```

当你写 `@CrewBase` 时，Python 会调用 `_CrewBaseType.__call__(CrewBase, MyCrew)`，它提取被装饰类的名称、基类和属性字典，然后用 `CrewBaseMeta` 重新创建这个类。这是一种经典的"通过装饰器应用元类"的技巧。

**第二层：`CrewBaseMeta` 元类**

`CrewBaseMeta` 是真正执行类变换的元类：

```python
class CrewBaseMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)

        cls.is_crew_class = True
        cls._crew_name = name

        # 执行类级别的初始化函数
        for setup_fn in _CLASS_SETUP_FUNCTIONS:
            setup_fn(cls)

        # 注入方法
        for method in _METHODS_TO_INJECT:
            setattr(cls, method.__name__, method)

        return cls
```

类创建时的三个 setup 函数：

```python
_CLASS_SETUP_FUNCTIONS = (
    _set_base_directory,    # 设置 base_directory（源文件所在目录）
    _set_config_paths,      # 设置 agents/tasks 配置文件路径
    _set_mcp_params,        # 设置 MCP Server 参数
)
```

注入到类中的方法：

```python
_METHODS_TO_INJECT = (
    close_mcp_server,
    get_mcp_tools,
    _load_config,
    load_configurations,
    staticmethod(load_yaml),
    map_all_agent_variables,
    _map_agent_variables,
    map_all_task_variables,
    _map_task_variables,
)
```

**第三层：实例创建拦截**

`CrewBaseMeta.__call__()` 拦截实例创建过程，在对象创建后执行初始化：

```python
def __call__(cls, *args, **kwargs):
    instance = super().__call__(*args, **kwargs)
    CrewBaseMeta._initialize_crew_instance(instance, cls)
    return instance
```

`_initialize_crew_instance()` 执行以下步骤：

1. **加载配置** — `instance.load_configurations()` 读取 YAML 文件
2. **收集方法** — `_get_all_methods(instance)` 获取所有非 dunder 方法
3. **映射 Agent 变量** — `map_all_agent_variables()` 解析 YAML 中的引用
4. **映射 Task 变量** — `map_all_task_variables()` 解析 YAML 中的引用
5. **构建元数据** — 创建 `CrewMetadata`，记录所有装饰过的方法
6. **注册 Hook** — `_register_crew_hooks()` 注册 LLM 和 Tool 的 Hook

### 22.2.2 CrewMetadata

`CrewMetadata` 是一个 TypedDict，存储了从装饰器收集到的所有方法信息：

```python
class CrewMetadata(TypedDict):
    original_methods: dict[str, Callable[..., Any]]
    original_tasks: dict[str, Callable[..., Task]]
    original_agents: dict[str, Callable[..., Agent]]
    before_kickoff: dict[str, Callable[..., Any]]
    after_kickoff: dict[str, Callable[..., Any]]
    kickoff: dict[str, Callable[..., Any]]
```

`_filter_methods()` 函数根据属性标记（如 `is_task`、`is_agent`）从所有方法中筛选出特定类别的方法：

```python
def _filter_methods(methods, attribute):
    return {
        name: method for name, method in methods.items()
        if hasattr(method, attribute)
    }
```

## 22.3 YAML 配置文件

### 22.3.1 agents.yaml

Agent 配置文件定义每个 Agent 的角色、目标和背景故事：

```yaml
researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher with a knack for uncovering
    the latest developments in {topic}.

reporting_analyst:
  role: >
    {topic} Reporting Analyst
  goal: >
    Create detailed reports based on {topic} data analysis
  backstory: >
    You're a meticulous analyst with a keen eye for detail.
```

YAML 中的 `{topic}` 是运行时变量占位符，在 `Crew.kickoff(inputs={"topic": "AI"})` 时替换。

`AgentConfig` TypedDict 定义了 YAML 中所有支持的字段：

```python
class AgentConfig(TypedDict, total=False):
    role: str
    goal: str
    backstory: str
    cache: bool
    verbose: bool
    llm: str                    # 可引用 @llm 装饰的方法名
    tools: list[str]            # 可引用 @tool 装饰的方法名
    function_calling_llm: str
    allow_code_execution: bool
    reasoning: bool
    multimodal: bool
    knowledge_sources: list[str]
    guardrail: Callable | str
    # ... 更多字段
```

### 22.3.2 tasks.yaml

Task 配置文件定义任务描述、期望输出和关联的 Agent：

```yaml
research_task:
  description: >
    Conduct a thorough research about {topic}
  expected_output: >
    A list with 10 bullet points of the most relevant information
  agent: researcher

reporting_task:
  description: >
    Review the context you got and expand each topic into a full section
  expected_output: >
    A fully fledged report with the main topics
  agent: reporting_analyst
```

`TaskConfig` TypedDict 定义了所有支持的字段：

```python
class TaskConfig(TypedDict, total=False):
    description: str
    expected_output: str
    agent: str                  # 引用 @agent 装饰的方法名
    context: list[str]          # 引用 @task 装饰的方法名列表
    tools: list[str]
    output_json: str            # 引用 @output_json 装饰的类名
    output_pydantic: str        # 引用 @output_pydantic 装饰的类名
    callback: str
    async_execution: bool
    human_input: bool
    guardrail: Callable | str
    # ... 更多字段
```

### 22.3.3 YAML 变量解析

当 YAML 配置中使用字符串引用（如 `agent: researcher` 或 `tools: [search_tool]`）时，`map_all_agent_variables()` 和 `map_all_task_variables()` 会将这些字符串解析为实际的 Python 对象。

Agent 变量映射：

```python
def map_all_agent_variables(self):
    llms = _filter_methods(self._all_methods, "is_llm")
    tool_functions = _filter_methods(self._all_methods, "is_tool")
    cache_handler_functions = _filter_methods(self._all_methods, "is_cache_handler")
    callbacks = _filter_methods(self._all_methods, "is_callback")

    for agent_name, agent_info in self.agents_config.items():
        self._map_agent_variables(agent_name, agent_info, llms,
                                  tool_functions, cache_handler_functions, callbacks)
```

具体映射逻辑：

```python
def _map_agent_variables(self, agent_name, agent_info, llms, ...):
    # 如果 llm 是字符串，尝试找到对应的 @llm 方法并调用
    if llm := agent_info.get("llm"):
        factory = llms.get(llm)
        self.agents_config[agent_name]["llm"] = factory() if factory else llm

    # 如果 tools 是字符串列表，找到对应的 @tool 方法并调用
    if tools := agent_info.get("tools"):
        if _is_string_list(tools):
            self.agents_config[agent_name]["tools"] = [
                tool_functions[tool]() for tool in tools
            ]
```

Task 变量映射类似，但还处理 `context`（Task 依赖）和 `output_json` / `output_pydantic`：

```python
def _map_task_variables(self, task_name, task_info, agents, tasks, ...):
    # context 列表中的任务名转为 Task 实例
    if context_list := task_info.get("context"):
        self.tasks_config[task_name]["context"] = [
            tasks[context_task_name]() for context_task_name in context_list
        ]

    # agent 字符串转为 Agent 实例
    if agent_name := task_info.get("agent"):
        self.tasks_config[task_name]["agent"] = agents[agent_name]()
```

## 22.4 装饰器详解

### 22.4.1 @agent

```python
def agent(meth: Callable[P, R]) -> AgentMethod[P, R]:
    """Marks a method as a crew agent."""
    return AgentMethod(memoize(meth))
```

`AgentMethod` 继承自 `DecoratedMethod`，设置 `is_agent = True` 标记。`memoize` 确保同一参数只创建一次 Agent 实例。

### 22.4.2 @task

```python
def task(meth: Callable[P, TaskResultT]) -> TaskMethod[P, TaskResultT]:
    """Marks a method as a crew task."""
    return TaskMethod(memoize(meth))
```

`TaskMethod` 与其他装饰器不同，它有特殊的 `ensure_task_name()` 逻辑：

```python
class TaskMethod(Generic[P, TaskResultT]):
    is_task: bool = True

    def ensure_task_name(self, result: TaskResultT) -> TaskResultT:
        if not result.name:
            result.name = self._meth.__name__
        return result
```

如果 Task 没有显式设置 `name`，会自动使用方法名作为 Task 名称。

`TaskMethod` 还使用了专门的 `BoundTaskMethod` 类来处理实例方法的绑定：

```python
class BoundTaskMethod(Generic[TaskResultT]):
    is_task: bool = True

    def __call__(self, *args, **kwargs) -> TaskResultT:
        result = self._task_method.unwrap()(self._obj, *args, **kwargs)
        result = _resolve_result(result)
        return self._task_method.ensure_task_name(result)
```

### 22.4.3 @crew

`@crew` 是最复杂的装饰器，它负责在调用时自动组装所有 Agent 和 Task：

```python
def crew(meth):
    @wraps(meth)
    def wrapper(self, *args, **kwargs):
        instantiated_tasks = []
        instantiated_agents = []
        agent_roles = set()

        # 按声明顺序实例化所有 Task
        tasks = self.__crew_metadata__["original_tasks"].items()
        for _, task_method in tasks:
            task_instance = _call_method(task_method, self)
            instantiated_tasks.append(task_instance)
            # 收集 Task 关联的 Agent
            agent_instance = getattr(task_instance, "agent", None)
            if agent_instance and agent_instance.role not in agent_roles:
                instantiated_agents.append(agent_instance)
                agent_roles.add(agent_instance.role)

        # 实例化未被 Task 引用的独立 Agent
        agents = self.__crew_metadata__["original_agents"].items()
        for _, agent_method in agents:
            agent_instance = _call_method(agent_method, self)
            if agent_instance.role not in agent_roles:
                instantiated_agents.append(agent_instance)
                agent_roles.add(agent_instance.role)

        self.agents = instantiated_agents
        self.tasks = instantiated_tasks

        # 调用用户定义的 crew 方法
        crew_instance = _call_method(meth, self, *args, **kwargs)

        # 注册 before/after kickoff 回调
        for hook_callback in self.__crew_metadata__["before_kickoff"].values():
            crew_instance.before_kickoff_callbacks.append(
                callback_wrapper(hook_callback, self))
        for hook_callback in self.__crew_metadata__["after_kickoff"].values():
            crew_instance.after_kickoff_callbacks.append(
                callback_wrapper(hook_callback, self))

        return crew_instance

    return memoize(wrapper)
```

这段代码的关键设计是：

1. **自动收集** — 遍历所有 `@task` 和 `@agent` 标记的方法，自动实例化
2. **去重** — 通过 `agent_roles` 集合确保同一 role 的 Agent 不会重复
3. **回调绑定** — `@before_kickoff` 和 `@after_kickoff` 方法自动绑定为 Crew 回调
4. **Memoize** — 整个 wrapper 被 memoize，确保多次调用 `.crew()` 返回同一实例

### 22.4.4 @before_kickoff / @after_kickoff

```python
def before_kickoff(meth):
    return BeforeKickoffMethod(meth)

def after_kickoff(meth):
    return AfterKickoffMethod(meth)
```

这两个装饰器标记的方法会在 Crew 的 `kickoff()` 前后执行。例如：

```python
@CrewBase
class MyCrew:
    @before_kickoff
    def prepare(self, inputs):
        inputs["timestamp"] = datetime.now().isoformat()
        return inputs

    @after_kickoff
    def cleanup(self, output):
        print(f"Crew completed with {len(output.tasks_output)} tasks")
        return output
```

### 22.4.5 其他装饰器

```python
@llm         # 标记为 LLM 工厂方法，YAML 中通过名称引用
@tool        # 标记为 Tool 工厂方法
@callback    # 标记为回调方法
@cache_handler # 标记为缓存处理器方法
@output_json    # 标记类为 JSON 输出格式
@output_pydantic # 标记类为 Pydantic 输出格式
```

## 22.5 Wrapper 类型体系

`wrappers.py` 定义了所有装饰器使用的 Wrapper 类，它们共享一个基类 `DecoratedMethod`：

```python
class DecoratedMethod(Generic[P, R]):
    def __init__(self, meth: Callable[P, R]) -> None:
        self._meth = meth
        _copy_method_metadata(self, meth)

    def __get__(self, obj, objtype=None):
        """Descriptor protocol 支持实例方法绑定"""
        if obj is None:
            return self
        inner = partial(self._meth, obj)

        def _bound(*args, **kwargs):
            result = _resolve_result(inner(*args, **kwargs))
            return result

        # 复制标记属性到绑定方法
        for attr in ("is_agent", "is_llm", "is_tool", ...):
            if hasattr(self, attr):
                setattr(_bound, attr, getattr(self, attr))
        return _bound
```

`DecoratedMethod` 实现了 descriptor protocol（`__get__`），确保作为类属性时能正确绑定到实例。同时，它将标记属性（如 `is_agent`）传递到绑定后的方法上，使得 `_filter_methods()` 能够正确识别。

继承层次：

```
DecoratedMethod
  ├── BeforeKickoffMethod (is_before_kickoff = True)
  ├── AfterKickoffMethod  (is_after_kickoff = True)
  ├── AgentMethod          (is_agent = True)
  ├── LLMMethod            (is_llm = True)
  ├── ToolMethod           (is_tool = True)
  ├── CallbackMethod       (is_callback = True)
  ├── CacheHandlerMethod   (is_cache_handler = True)
  └── CrewMethod           (is_crew = True)

TaskMethod (独立实现，不继承 DecoratedMethod)
  └── BoundTaskMethod (绑定到实例的 Task 方法)

OutputClass
  ├── OutputJsonClass    (is_output_json = True)
  └── OutputPydanticClass (is_output_pydantic = True)
```

`OutputClass` 不继承 `DecoratedMethod`，因为它装饰的是类而非方法：

```python
class OutputClass(Generic[T]):
    def __init__(self, cls: type[T]) -> None:
        self._cls = cls

    def __call__(self, *args, **kwargs) -> T:
        return self._cls(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._cls, name)
```

## 22.6 memoize 缓存机制

`utils.py` 中的 `memoize` 函数用于缓存装饰器方法的返回值：

```python
def memoize(meth: Callable[P, R]) -> Callable[P, R]:
    if inspect.iscoroutinefunction(meth):
        return cast(Callable[P, R], _memoize_async(meth))
    return _memoize_sync(meth)
```

同步版本的实现：

```python
def _memoize_sync(meth):
    @wraps(meth)
    def wrapper(*args, **kwargs):
        hashable_args = tuple(_make_hashable(arg) for arg in args)
        hashable_kwargs = tuple(
            sorted((k, _make_hashable(v)) for k, v in kwargs.items()))
        cache_key = str((hashable_args, hashable_kwargs))

        cached_result = cache.read(tool=meth.__name__, input=cache_key)
        if cached_result is not None:
            return cached_result

        result = meth(*args, **kwargs)
        cache.add(tool=meth.__name__, input=cache_key, output=result)
        return result
    return wrapper
```

`_make_hashable()` 处理不同类型参数的哈希化：

```python
def _make_hashable(arg):
    if isinstance(arg, BaseModel):
        return arg.model_dump_json()       # Pydantic 模型转 JSON
    if isinstance(arg, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in arg.items()))
    if isinstance(arg, list):
        return tuple(_make_hashable(item) for item in arg)
    if hasattr(arg, "__dict__"):
        return ("__instance__", id(arg))    # 对象用 id 作为键
    return arg
```

使用 `CacheHandler` 作为全局缓存后端。这确保了在同一个 Crew 生命周期内，相同参数的 `@agent` 或 `@task` 方法只执行一次——这对于 YAML 配置中多处引用同一个 Agent 的场景至关重要。

## 22.7 Hook 注册机制

`_register_crew_hooks()` 在实例初始化时检测并注册 LLM 和 Tool 的 Hook 方法：

```python
def _register_crew_hooks(instance, cls):
    hook_methods = {
        name: method
        for name, method in cls.__dict__.items()
        if any(hasattr(method, attr) for attr in [
            "is_before_llm_call_hook",
            "is_after_llm_call_hook",
            "is_before_tool_call_hook",
            "is_after_tool_call_hook",
        ])
    }
```

Hook 支持按 Agent 或 Tool 名称过滤：

```python
if has_agent_filter:
    agents_filter = hook_method._filter_agents

    def make_filtered_before_llm(bound_fn, agents_list):
        def filtered(context):
            if context.agent and context.agent.role not in agents_list:
                return None
            return bound_fn(context)
        return filtered
```

这允许开发者在 Crew 类中声明式地定义 Hook，并通过装饰器参数限定作用范围。

## 22.8 MCP Server 集成

`@CrewBase` 还支持 MCP (Model Context Protocol) Server 的集成：

```python
def _set_mcp_params(cls):
    cls.mcp_server_params = getattr(cls, "mcp_server_params", None)
    cls.mcp_connect_timeout = getattr(cls, "mcp_connect_timeout", 30)
```

注入的 `get_mcp_tools()` 方法用于获取 MCP Server 提供的工具：

```python
def get_mcp_tools(self, *tool_names):
    if not self.mcp_server_params:
        return []

    from crewai_tools import MCPServerAdapter

    if self._mcp_server_adapter is None:
        self._mcp_server_adapter = MCPServerAdapter(
            self.mcp_server_params,
            connect_timeout=self.mcp_connect_timeout
        )

    return self._mcp_server_adapter.tools.filter_by_names(tool_names or None)
```

`close_mcp_server()` 被自动注册为 `after_kickoff` 回调，确保 MCP 连接在 Crew 执行完毕后关闭。

## 22.9 项目模板详解

### 22.9.1 模板 crew.py

CLI 生成的 `crew.py` 展示了 `@CrewBase` 的标准用法：

```python v-pre
@CrewBase
class {{crew_name}}():
    """{{crew_name}} crew"""
    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
```

注意 `self.agents_config` 和 `self.tasks_config` 是由 `CrewBaseMeta` 自动注入的——它们是从 YAML 文件加载的字典。

### 22.9.2 模板 main.py

`main.py` 提供了所有 CLI 命令对应的入口函数：

```python v-pre
def run():
    inputs = {'topic': 'AI LLMs', 'current_year': str(datetime.now().year)}
    {{crew_name}}().crew().kickoff(inputs=inputs)

def train():
    inputs = {"topic": "AI LLMs", 'current_year': str(datetime.now().year)}
    {{crew_name}}().crew().train(
        n_iterations=int(sys.argv[1]),
        filename=sys.argv[2],
        inputs=inputs
    )

def replay():
    {{crew_name}}().crew().replay(task_id=sys.argv[1])

def test():
    inputs = {"topic": "AI LLMs", "current_year": str(datetime.now().year)}
    {{crew_name}}().crew().test(
        n_iterations=int(sys.argv[1]),
        eval_llm=sys.argv[2],
        inputs=inputs
    )
```

每个函数都创建一个新的 Crew 类实例，调用 `.crew()` 获取 Crew 对象，然后执行对应操作。这遵循了 `@CrewBase` 的设计：Crew 类是 Crew 的工厂。

### 22.9.3 pyproject.toml 脚本映射

这些函数通过 pyproject.toml 注册为命令行脚本：

```toml
[project.scripts]
run_crew = "my_crew.main:run"
train = "my_crew.main:train"
replay = "my_crew.main:replay"
test = "my_crew.main:test"
run_with_trigger = "my_crew.main:run_with_trigger"
```

这就是 CLI 的 `uv run run_crew` 能够正确调用到用户代码的原因。

## 22.10 项目目录约定

CrewAI 项目遵循标准的目录约定：

```
my_project/                    # 项目根目录
├── pyproject.toml            # 项目配置，包含 [tool.crewai] 段
├── .env                      # 环境变量（API keys 等）
├── .gitignore
├── AGENTS.md                 # AI 助手指南文件
├── README.md
├── knowledge/                # Knowledge 文件目录
│   └── user_preference.txt
├── tests/                    # 测试目录
└── src/
    └── my_project/           # Python 包目录
        ├── __init__.py
        ├── main.py           # 入口函数
        ├── crew.py           # @CrewBase Crew 定义
        ├── config/
        │   ├── agents.yaml   # Agent 配置
        │   └── tasks.yaml    # Task 配置
        └── tools/
            ├── __init__.py
            └── custom_tool.py
```

对于 Flow 项目，结构略有不同：

```
my_flow/
├── pyproject.toml            # type = "flow"
├── .env
└── src/
    └── my_flow/
        ├── __init__.py
        ├── main.py
        ├── tools/
        └── crews/            # Flow 内的 Crew 子项目
            └── poem_crew/
                ├── crew.py
                └── config/
                    ├── agents.yaml
                    └── tasks.yaml
```

Flow 项目中的每个 Crew 是一个子模块，不包含独立的 `pyproject.toml` 或 `main.py`。

## 22.11 YAML 配置加载流程

完整的配置加载流程如下：

```
1. CrewBaseMeta.__new__()
   └─> _set_config_paths(cls)
       └─> cls.original_agents_config_path = "config/agents.yaml"

2. CrewBaseMeta.__call__() (实例创建)
   └─> _initialize_crew_instance(instance, cls)
       └─> instance.load_configurations()
           └─> self._load_config("config/agents.yaml", "agent")
               └─> load_yaml(base_directory / "config/agents.yaml")
                   └─> yaml.safe_load(file)

3. instance.map_all_agent_variables()
   └─> 遍历 agents_config，解析 string 引用为实际对象

4. instance.map_all_task_variables()
   └─> 遍历 tasks_config，解析 string 引用为实际对象
```

如果配置文件不存在，会打印警告但不会报错——这支持纯代码方式定义 Agent 和 Task。

## 22.12 实验框架（experimental）

`experimental/` 目录包含评估框架，目前最主要的功能是 `evaluation/` 子目录。

### 22.12.1 ExperimentRunner

`ExperimentRunner` 用于系统化地测试 Crew 或 Agent 的表现：

```python
class ExperimentRunner:
    def __init__(self, dataset: list[dict[str, Any]]):
        self.dataset = dataset
        self.evaluator = None

    def run(self, crew=None, agents=None, print_summary=False):
        if crew and not agents:
            agents = crew.agents
        self.evaluator = create_default_evaluator(agents=agents)

        results = []
        for test_case in self.dataset:
            self.evaluator.reset_iterations_results()
            result = self._run_test_case(test_case, crew=crew, agents=agents)
            results.append(result)
        return ExperimentResults(results)
```

### 22.12.2 评估指标

评估框架在 `metrics/` 目录中定义了多个维度的指标：

- **goal_metrics.py** — 评估 Agent 是否达成目标
- **reasoning_metrics.py** — 评估推理质量
- **semantic_quality_metrics.py** — 评估输出的语义质量
- **tools_metrics.py** — 评估工具使用的有效性

### 22.12.3 测试断言

`testing.py` 提供了与测试框架集成的断言函数：

```python
def run_experiment(dataset, crew=None, agents=None, verbose=False):
    runner = ExperimentRunner(dataset=dataset)
    return runner.run(agents=agents, crew=crew, print_summary=verbose)

def assert_experiment_successfully(experiment_results, baseline_filepath=None):
    failed_tests = [r for r in experiment_results.results if not r.passed]
    if failed_tests:
        raise AssertionError(f"The following test cases failed: ...")

    comparison = experiment_results.compare_with_baseline(
        baseline_filepath=baseline_filepath)
    assert_experiment_no_regression(comparison)
```

`assert_experiment_no_regression()` 检测回归——如果之前通过的测试现在失败了，会抛出 `AssertionError`。

## 22.13 异步支持

project 模块在多个层面支持异步：

1. **`_resolve_result()`** — 自动处理协程返回值
2. **`_memoize_async()`** — 异步方法的 memoize 实现
3. **`_call_method()`** — 统一处理同步/异步方法调用

```python
def _call_method(method, *args, **kwargs):
    result = method(*args, **kwargs)
    if inspect.iscoroutine(result):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, result).result()
        return asyncio.run(result)
    return result
```

这确保了即使在同步上下文中使用 `async def` 定义的 `@agent` 或 `@task` 方法，也能正确执行。

## 本章要点

- `@CrewBase` 通过 `_CrewBaseType` 和 `CrewBaseMeta` 两层元类机制，将普通 Python 类转变为声明式的 Crew 定义
- `CrewBaseMeta.__new__()` 在类创建时设置基础目录、配置路径，并注入方法；`__call__()` 在实例创建时加载 YAML 配置并映射变量引用
- YAML 配置文件（`agents.yaml` / `tasks.yaml`）中的字符串引用会被自动解析为对应的 Python 对象（Agent、Task、Tool、LLM 等）
- `@agent`、`@task`、`@crew` 等装饰器基于 `DecoratedMethod` Wrapper 类，通过 descriptor protocol 支持实例方法绑定
- `@task` 装饰器会自动用方法名填充未设置的 Task name
- `@crew` 装饰器在调用时自动收集所有 `@task` 和 `@agent` 方法的实例，并注册 `@before_kickoff` / `@after_kickoff` 回调
- `memoize` 机制确保同一参数的装饰器方法只执行一次，使用 `CacheHandler` 作为后端
<span v-pre>- 项目模板通过占位符替换（`{{crew_name}}`、`{{folder_name}}`）生成标准的 Python 项目结构</span>
- `experimental/evaluation` 提供了系统化的实验框架，支持多指标评估、基线比较和回归检测
- Hook 注册机制支持在 Crew 类中声明式地定义 LLM 和 Tool 的拦截器，并支持按 Agent 或 Tool 名称过滤
