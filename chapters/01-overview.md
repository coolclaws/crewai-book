# 第 1 章　项目概览与设计哲学

> CrewAI 是一个多 Agent 协作编排框架——它不直接执行推理，而是编排多个 Agent 协同完成复杂任务。

## 1.1 CrewAI 是什么

CrewAI（截至 v1.10.1）是一个用 Python 编写的开源框架，核心使命是让开发者能以**声明式**的方式定义一组 AI Agent，赋予它们各自的角色、目标和背景故事，然后把一系列 Task 分配给这些 Agent，由框架负责调度执行。

打开 `__init__.py`，我们可以看到框架对外暴露的核心类一目了然：

```python
# crewai/__init__.py（精简）
from crewai.agent.core import Agent
from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.flow.flow import Flow
from crewai.knowledge.knowledge import Knowledge
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.process import Process
from crewai.task import Task
from crewai.tasks.llm_guardrail import LLMGuardrail
from crewai.tasks.task_output import TaskOutput

__version__ = "1.10.1"

__all__ = [
    "LLM", "Agent", "BaseLLM", "Crew", "CrewOutput",
    "Flow", "Knowledge", "LLMGuardrail", "Memory",
    "Process", "Task", "TaskOutput", "__version__",
]
```

这 12 个公开名称构成了 CrewAI 的 **公共 API 面**。其中有四个概念是整个框架的骨架：

| 概念 | 类 | 职责 |
|------|------|------|
| 团队 | `Crew` | 编排 Agent 和 Task 的容器，负责调度执行 |
| 成员 | `Agent` | 拥有角色、目标和工具的执行单元 |
| 任务 | `Task` | 描述需要完成的工作，包含期望输出 |
| 工作流 | `Flow` | 将多个 Crew 编排成有向图，支持条件路由 |

值得注意的是 `Memory` 使用了 lazy import 模式——因为它依赖 lancedb 等重型库，只有在首次访问时才会加载：

```python
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Memory": ("crewai.memory.unified_memory", "Memory"),
}

def __getattr__(name: str) -> Any:
    """Lazily import heavy modules (e.g. Memory → lancedb) on first access."""
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'crewai' has no attribute {name!r}")
```

这是一个典型的 **延迟加载** 技巧：首次 `import crewai` 的速度不会因为 lancedb/chromadb 等向量数据库而变慢。

## 1.2 四大核心抽象

### 1.2.1 Crew——团队

`Crew` 是 CrewAI 的最高层编排单元。它继承自 Pydantic `BaseModel` 和 `FlowTrackable`，将 Agent、Task、Process（执行策略）组合在一起：

```python
# crewai/crew.py
class Crew(FlowTrackable, BaseModel):
    """
    Represents a group of agents, defining how they should collaborate
    and the tasks they should perform.
    """

    name: str | None = Field(default="crew")
    cache: bool = Field(default=True)
    tasks: list[Task] = Field(default_factory=list)
    agents: list[BaseAgent] = Field(default_factory=list)
    process: Process = Field(default=Process.sequential)
    verbose: bool = Field(default=False)
    memory: bool | Any = Field(
        default=False,
        description=(
            "Enable crew memory. Pass True for default Memory(), "
            "or a Memory/MemoryScope/MemorySlice instance for custom configuration."
        ),
    )
    planning: bool = Field(default=False)
    planning_llm: str | InstanceOf[BaseLLM] | Any | None = Field(default=None)
    security_config: SecurityConfig = Field(default_factory=SecurityConfig)
```

几个关键设计决策：

1. **`process` 字段** 决定了任务执行的策略。`Process` 是一个简单的枚举：

```python
# crewai/process.py
class Process(str, Enum):
    sequential = "sequential"
    hierarchical = "hierarchical"
    # TODO: consensual = 'consensual'
```

`sequential` 模式下，Task 按列表顺序依次执行，前一个 Task 的输出自动作为后一个 Task 的上下文；`hierarchical` 模式下，框架会引入一个 manager Agent 来决定任务分配。

2. **`FlowTrackable` mixin** 让 Crew 可以被 Flow 编排——当 Crew 作为 Flow 中的一个节点时，Flow 能追踪其执行状态。

3. **memory 字段的多态设计** 接受 `bool` 或实际的 Memory 实例——`True` 表示使用默认配置，也可以传入自定义的 Memory 对象。

### 1.2.2 Agent——成员

Agent 是 CrewAI 中最能体现"角色驱动"设计哲学的类。它的三大核心字段 `role`、`goal`、`backstory` 构成了 Agent 的"人设"：

```python
# crewai/agents/agent_builder/base_agent.py
class BaseAgent(BaseModel, ABC, metaclass=AgentMeta):
    """Abstract Base Class for all third party agents compatible with CrewAI."""

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Objective of the agent")
    backstory: str = Field(description="Backstory of the agent")
    cache: bool = Field(default=True)
    verbose: bool = Field(default=False)
    max_rpm: int | None = Field(default=None)
    allow_delegation: bool = Field(default=False)
    tools: list[Any] | None = Field(default_factory=list)
    max_iter: int = Field(default=25)
```

`BaseAgent` 是一个抽象基类，使用了 `AgentMeta` 元类。真正供用户使用的是 `Agent`（位于 `crewai/agent/core.py`），它在 `BaseAgent` 基础上增加了大量实用字段：

```python
# crewai/agent/core.py
class Agent(BaseAgent):
    """Represents an agent in a system.

    Each agent has a role, a goal, a backstory, and an optional language model (llm).
    The agent can also have memory, can operate in verbose mode, and can delegate
    tasks to other agents.
    """

    llm: str | InstanceOf[BaseLLM] | Any = Field(
        description="Language model that will run the agent.", default=None
    )
    function_calling_llm: str | InstanceOf[BaseLLM] | Any | None = Field(default=None)
    allow_code_execution: bool | None = Field(default=False)
    respect_context_window: bool = Field(default=True)
    max_retry_limit: int = Field(default=2)
    reasoning: bool = Field(default=False)
    code_execution_mode: Literal["safe", "unsafe"] = Field(default="safe")
    use_system_prompt: bool | None = Field(default=True)
```

注意 `llm` 字段的类型是 `str | InstanceOf[BaseLLM] | Any`——你可以传入模型名称字符串（如 `"gpt-4o"`），也可以传入 `LLM` 实例。框架内部会通过 `create_llm()` 工具函数统一转换。

`reasoning` 字段是一个较新的功能标志——启用后，Agent 会在执行 Task 前先进行"思考和规划"，类似 Chain-of-Thought 的推理过程。

### 1.2.3 Task——任务

`Task` 类描述了一项具体的工作：

```python
# crewai/task.py
class Task(BaseModel):
    """Class that represents a task to be executed.

    Each task must have a description, an expected output and an agent
    responsible for execution.
    """

    description: str = Field(description="Description of the actual task.")
    expected_output: str = Field(
        description="Clear definition of expected output for the task."
    )
    agent: BaseAgent | None = Field(default=None)
    context: list[Task] | None | _NotSpecified = Field(default=NOT_SPECIFIED)
    async_execution: bool | None = Field(default=False)
    output_json: type[BaseModel] | None = Field(default=None)
    output_pydantic: type[BaseModel] | None = Field(default=None)
    response_model: type[BaseModel] | None = Field(default=None)
    output_file: str | None = Field(default=None)
    tools: list[BaseTool] | None = Field(default_factory=list)
    human_input: bool | None = Field(default=False)
    guardrail: GuardrailType | None = Field(default=None)
    guardrails: GuardrailsType | None = Field(default=None)
    security_config: SecurityConfig = Field(default_factory=SecurityConfig)
```

Task 的设计有几个亮点：

- **`description` + `expected_output`**：这是 CrewAI 的核心理念——任务不只是一句话，还必须明确"期望输出是什么"。这迫使用户在设计时就想清楚每个任务的目标。
- **`context` 字段**：默认值是一个特殊的 `NOT_SPECIFIED` 哨兵值，而不是 `None`。这样框架可以区分"用户未设置 context"和"用户显式设置为空"——前者会自动使用上一个 Task 的输出作为上下文，后者则不会。
- **结构化输出**：`output_json`、`output_pydantic`、`response_model` 三种方式让 Task 输出可以被 Pydantic model 约束。
- **guardrail 机制**：通过 `guardrail` / `guardrails` 字段，可以在 Task 输出后执行验证逻辑，确保输出质量。如果验证失败，框架会要求 Agent 重新执行。
- **`input_files`**：支持向 Task 传入文件，框架会自动处理文件内容的提取和注入。

### 1.2.4 Flow——工作流

`Flow` 是 CrewAI 中最复杂的抽象，它将多个方法（通常包含 Crew 调用）编排成一个有向图：

```python
# crewai/flow/flow.py
class Flow(Generic[T], metaclass=FlowMeta):
    """Base class for all flows.

    type parameter T must be either dict[str, Any] or a subclass of BaseModel.
    """

    _start_methods: ClassVar[list[FlowMethodName]] = []
    _listeners: ClassVar[dict[FlowMethodName, SimpleFlowCondition | FlowCondition]] = {}
    _routers: ClassVar[set[FlowMethodName]] = set()
    _router_paths: ClassVar[dict[FlowMethodName, list[FlowMethodName]]] = {}
    initial_state: type[T] | T | None = None
```

Flow 使用三个装饰器来定义执行图：

```python
@start()                    # 标记入口方法
def generate_topics(self):
    ...

@listen(generate_topics)    # 监听 generate_topics 完成后执行
def research_topic(self, topics):
    ...

@router(research_topic)     # 根据返回值路由到不同分支
def decide_next(self, result):
    if result.quality > 0.8:
        return "publish"
    return "revise"
```

`@start`、`@listen`、`@router` 三个装饰器通过 `FlowMeta` 元类在类定义时被收集，构建出一张方法依赖图。运行时，Flow 按照这张图的拓扑顺序调度方法执行。

Flow 还支持：

- **泛型状态管理**：`Flow[MyState]` 可以持有类型安全的状态对象
- **持久化**：通过 `FlowPersistence` 接口（内置 SQLite 实现）保存和恢复执行状态
- **AND/OR 条件**：`@listen(and_("method_a", "method_b"))` 表示两个方法都完成后才触发
- **人类反馈**：`self.ask()` 方法可以在 Flow 执行中暂停等待人类输入
- **可视化**：内置的 visualization 模块可以将 Flow 渲染为交互式图表

## 1.3 设计哲学：角色驱动与任务导向

CrewAI 的设计哲学可以用两个词概括：**角色驱动**（Role-Driven）和**任务导向**（Task-Oriented）。

### 角色驱动

CrewAI 借鉴了现实世界中团队协作的隐喻：每个 Agent 有明确的"角色"（role）、"目标"（goal）和"背景故事"（backstory）。这三个字段不仅是描述性的——它们会被注入到 LLM 的 system prompt 中，直接影响 Agent 的行为模式。

```python
# 用户代码示例
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="You work at a leading tech think tank...",
    tools=[search_tool],
    llm="gpt-4o",
)
```

这与传统的 prompt engineering 不同——开发者不需要手写完整的 system prompt，而是通过结构化的角色定义来"塑造" Agent 的行为。框架内部的 `Prompts` 类负责将这些字段组装成最终的 prompt。

### 任务导向

Agent 不会自主决定做什么——它们的行为完全由 Task 驱动。每个 Task 明确描述了"做什么"（description）和"做成什么样"（expected_output）。这种设计有两个好处：

1. **可预测性**：相比让 Agent 自由探索，任务导向的设计让执行过程更可控
2. **可组合性**：Task 之间通过 `context` 字段形成数据流，容易构建复杂的工作链

### 执行模型

Crew 的执行模型分为两种：

- **Sequential（顺序执行）**：Task 按列表顺序依次执行，适合线性工作流
- **Hierarchical（层级执行）**：引入 manager Agent 来动态分配 Task，适合需要灵活决策的场景

Flow 则在 Crew 之上提供了更高层的编排——多个 Crew 可以作为 Flow 中的方法，通过 `@listen` 和 `@router` 装饰器构建复杂的工作流图。

## 1.4 与 LangGraph 的对比

CrewAI 和 LangGraph 都是 Agent 编排框架，但定位有明显差异：

| 维度 | CrewAI | LangGraph |
|------|--------|-----------|
| **抽象层级** | 高层，面向业务 | 底层，面向开发者 |
| **核心隐喻** | 团队协作（Crew/Agent/Task） | 有向图（Node/Edge/State） |
| **Agent 定义** | 角色驱动（role/goal/backstory） | 函数式（自定义 node 函数） |
| **任务分配** | 声明式（Task 绑定 Agent） | 命令式（graph 定义路由） |
| **状态管理** | Crew 层面自动传递 context | 显式定义 State schema |
| **学习曲线** | 较低，概念直观 | 较高，需要理解图的概念 |
| **灵活性** | 中等，框架提供约定 | 高，几乎完全自定义 |
| **适用场景** | 业务流程自动化、内容生产 | 复杂推理、自定义控制流 |

简单来说，CrewAI 更像是"AI 团队管理工具"——你定义团队成员（Agent）和工作任务（Task），框架帮你管理协作过程。而 LangGraph 更像是"AI 工作流引擎"——你直接定义执行图的每个节点和边。

如果你的需求是"让几个 AI Agent 协作完成一组明确的任务"，CrewAI 的抽象层级更合适；如果你需要精细控制每一步的执行逻辑，LangGraph 提供了更大的灵活性。

CrewAI 的 Flow 机制在一定程度上弥补了灵活性的差距——它允许开发者定义方法级别的有向图，但整体设计仍然保持"业务优先"的理念。

## 1.5 技术栈概览

CrewAI 的技术选型值得一提：

- **Pydantic v2**：所有核心类（Crew、Agent、Task）都继承自 `BaseModel`，利用 Pydantic 的验证、序列化能力
- **OpenTelemetry**：内置 tracing 支持，Crew 和 Flow 的执行可以被追踪和可视化
- **事件系统**：自定义的 `EventBus` 和 `EventListener` 机制，支持松耦合的组件通信
- **LLM 抽象层**：通过 `LLM` 和 `BaseLLM` 类封装不同的模型提供商（OpenAI、Anthropic、Google 等）
- **MCP 支持**：原生支持 Model Context Protocol，Agent 可以直接使用 MCP server 提供的工具
- **A2A 协议**：通过 Agent-to-Agent 协议支持跨框架的 Agent 通信

`__init__.py` 中还有一个有意思的细节——telemetry。CrewAI 在 import 时会通过后台线程向 Scarf 发送一次安装追踪请求（可以通过环境变量禁用）：

```python
def _track_install_async() -> None:
    """Track installation in background thread to avoid blocking imports."""
    if not Telemetry._is_telemetry_disabled():
        thread = threading.Thread(target=_track_install, daemon=True)
        thread.start()

_track_install_async()
```

这种做法在开源项目中比较常见，用于统计安装量。注意它使用了 daemon thread，不会阻塞主进程退出。

## 本章要点

- CrewAI 定位为**多 Agent 协作编排框架**，核心职责是编排而非推理执行
- 四大核心抽象：**Crew**（团队编排）、**Agent**（角色驱动的执行单元）、**Task**（描述+期望输出）、**Flow**（有向图工作流）
- Agent 的设计采用**角色驱动**模式：`role` / `goal` / `backstory` 三元组定义 Agent 的"人设"
- Task 采用**任务导向**设计：每个任务必须明确 `description` 和 `expected_output`
- Process 支持 `sequential`（顺序）和 `hierarchical`（层级）两种执行策略
- Flow 通过 `@start`、`@listen`、`@router` 三个装饰器构建方法依赖图
- 与 LangGraph 相比，CrewAI 抽象层级更高、更面向业务场景
- 技术栈基于 Pydantic v2、OpenTelemetry、自定义事件系统
- Memory 使用 lazy import 避免重型依赖拖慢导入速度

## 延伸阅读

- [CrewAI 核心概念](https://docs.crewai.com/concepts/crews)
- [Agent 如何工作](https://docs.crewai.com/concepts/agents)
- [Flow 概述](https://docs.crewai.com/concepts/flows)
