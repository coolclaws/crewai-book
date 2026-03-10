# 附录 C：名词解释（Glossary）

本附录收录了本书中出现的关键术语，按英文字母顺序排列。每个条目包含英文术语和中文解释。

---

**Agent** — 智能代理。CrewAI 中的核心执行单元，封装了角色（role）、目标（goal）、背景故事（backstory）以及所使用的 LLM 和工具。Agent 接收 Task 并自主决定如何完成。

**Agent Executor** — Agent 执行器。负责 Agent 内部的"思考-行动-观察"循环（ReAct loop），管理 LLM 调用、工具执行和结果解析的底层引擎。

**Backstory** — 背景故事。Agent 配置中的一段文本，为 LLM 提供角色的背景信息和行为风格指导，帮助 Agent 生成更符合预期的输出。

**Callback** — 回调函数。在特定事件（如 Task 完成、Crew kickoff 前后）触发的函数，用于实现自定义的前处理和后处理逻辑。

**Chunk / Chunking** — 分块。将长文本切分为较小片段的过程，常用于 Knowledge 和 Memory 系统中，以适应 Embedding 模型的输入长度限制。

**Click** — Python 命令行框架。CrewAI CLI 基于 Click 构建，通过装饰器定义命令、参数和选项。

**Context** — 上下文。在 Task 配置中，context 指定当前任务依赖的前置任务列表。前置任务的输出会作为上下文传递给当前任务。

**Context Window** — 上下文窗口。LLM 能够处理的最大 Token 数量。CrewAI 通过 `respect_context_window` 参数自动管理上下文长度，避免超出限制。

**Crew** — 团队。CrewAI 中的编排容器，管理一组 Agent 和 Task 的协作执行。支持 sequential（顺序）和 hierarchical（层级）两种 Process 模式。

**CrewBase** — Crew 基类装饰器。`@CrewBase` 将普通 Python 类转变为声明式的 Crew 定义，通过元类机制自动加载 YAML 配置并映射变量引用。

**Delegation** — 委派。Agent 将任务委托给其他 Agent 执行的能力。通过 `allow_delegation` 参数控制，在 hierarchical 模式中由 Manager Agent 协调。

**Embedder** — 嵌入引擎。将文本转换为向量表示的组件，用于 Memory 和 Knowledge 系统中的语义搜索。支持 OpenAI、Google、Cohere、Ollama 等多种 Provider。

**Entity Memory** — 实体记忆。Memory 系统的一种类型，专门存储和检索关于特定实体（人、地点、概念等）的信息。

**Event** — 事件。CrewAI 事件系统中的消息单元，包含类型、时间戳和数据。用于实现组件间的松耦合通信和可观测性。

**Expected Output** — 期望输出。Task 配置中的字段，描述任务完成后应产生的输出格式和内容，指导 Agent 生成符合要求的结果。

**Experiment** — 实验。experimental 模块中的测试框架概念，通过定义数据集和期望分数来系统化地评估 Crew 的表现。

**Flow** — 工作流。CrewAI 中的高级编排系统，支持有状态的、多步骤的执行链。通过 `@start`、`@listen`、`@router` 装饰器定义执行图。

**Flow State** — Flow 状态。Flow 实例中维护的全局状态对象，在各个步骤间共享和修改。可以是简单字典或 Pydantic BaseModel。

**Function Calling** — 函数调用。LLM 的一种能力，允许模型通过结构化输出来调用外部函数（即 Tool）。CrewAI 的工具系统基于此能力构建。

**Guardrail** — 安全护栏。对 Agent 或 Task 输出进行验证的机制。返回 `(True, result)` 表示通过，`(False, error)` 表示需要重试。支持最大重试次数限制。

**Hierarchical Process** — 层级流程。Crew 的一种执行模式，由 Manager Agent 统筹安排任务分配，其他 Agent 按指令执行。

**Hook** — 钩子。在 LLM 调用或 Tool 执行前后插入的拦截函数，用于修改输入/输出或添加额外逻辑。支持按 Agent 和 Tool 名称过滤。

**Human Input** — 人工输入。Task 配置中的选项，当启用时 Agent 会在执行过程中暂停并请求人工确认或补充信息。

**Kickoff** — 启动执行。`Crew.kickoff()` 是 Crew 的主要执行方法，接收输入参数并驱动所有 Task 的执行。

**Knowledge** — 知识。外部信息源的集成系统，支持从文件、字符串、URL 等来源加载知识，通过向量搜索在 Agent 执行时提供相关信息。

**Knowledge Source** — 知识源。Knowledge 系统中的数据提供者，如 `StringKnowledgeSource`、`FileKnowledgeSource` 等，负责加载和预处理知识数据。

**LanceDB** — 开源向量数据库。CrewAI Memory 系统的默认向量存储后端，支持高效的相似度搜索。

**litellm** — LLM 统一接口库。CrewAI 的 `LLM` 类基于 litellm 构建，提供对 100+ LLM Provider 的统一访问。

**Long-Term Memory** — 长期记忆。跨多次 Crew 运行持久化的记忆，存储在 SQLite 或向量数据库中。

**Manager Agent** — 管理者 Agent。在 hierarchical Process 模式中负责任务分配和协调的特殊 Agent。

**MCP (Model Context Protocol)** — 模型上下文协议。标准化的 AI 工具协议，CrewAI 通过 `MCPServerAdapter` 支持连接 MCP Server 并使用其提供的工具。

**Memoize** — 记忆化。缓存函数调用结果的优化技术。在 project 模块中用于确保 `@agent` 和 `@task` 方法对相同参数只执行一次。

**Memory** — 记忆。Agent 在执行过程中积累的信息存储系统，分为短期记忆、长期记忆、实体记忆等类型。

**Metaclass** — 元类。Python 中"类的类"，用于控制类的创建过程。`CrewBaseMeta` 通过元类机制在类创建时注入配置加载和方法映射功能。

**Pipeline** — 管道。多个 Crew 的串联执行机制，前一个 Crew 的输出作为后一个 Crew 的输入。支持并行阶段。

**Planning** — 计划。Crew 执行前的自动任务规划功能，通过 LLM 分析任务列表并生成执行计划。

**Process** — 流程。Crew 中 Task 的执行策略，`sequential` 按顺序执行，`hierarchical` 由 Manager Agent 分配执行。

**Provider** — 提供者。LLM 服务的供应商，如 OpenAI、Anthropic、Google、Azure 等。CrewAI 通过 litellm 支持多种 Provider。

**RAG (Retrieval-Augmented Generation)** — 检索增强生成。结合向量搜索和 LLM 生成的技术。CrewAI 的 Knowledge 和 Memory 系统本质上是 RAG 架构。

**ReAct** — Reasoning and Acting。一种 Agent 推理模式，交替进行"思考"和"行动"步骤。CrewAI 的 Agent Executor 实现了这一模式。

**Replay** — 重放。从之前执行的某个 Task 开始重新执行 Crew 的能力，利用 SQLite 存储的历史 Task 输出跳过已完成的任务。

**Role** — 角色。Agent 的核心标识属性，描述 Agent 在 Crew 中扮演的角色（如"资深研究员"、"数据分析师"）。

**Router** — 路由器。Flow 中的条件分支机制，通过 `@router` 装饰器根据条件将执行导向不同的下游方法。

**Scope** — 作用域。Memory 系统中的命名空间概念，不同 scope 下的 Memory 数据相互隔离。

**Sequential Process** — 顺序流程。Crew 的默认执行模式，Task 按定义顺序依次执行，前一个 Task 的输出作为后一个 Task 的上下文。

**Short-Term Memory** — 短期记忆。仅在单次 Crew 运行期间有效的记忆，运行结束后不持久化。

**State** — 状态。Flow 中在各步骤间共享的数据对象，可以是非结构化字典或 Pydantic 模型。

**Storage** — 存储。Memory 和 Knowledge 的持久化后端，包括 LanceDB（向量存储）、SQLite（结构化存储）等。

**Task** — 任务。Agent 需要完成的具体工作单元，包含描述、期望输出、关联 Agent 和上下文依赖等配置。

**Telemetry** — 遥测。CrewAI 的使用数据收集机制，用于改进框架。可通过环境变量或 CLI 命令控制开关。

<span v-pre>**Template** — 模板。CLI 创建项目时使用的文件模板，包含占位符（如 `{{crew_name}}`），在创建时替换为实际值。</span>

**Token** — 令牌。LLM 处理文本的基本单位。CrewAI 通过 `token_usage` 跟踪 Token 消耗，并通过 `max_tokens` 控制生成长度。

**Tool** — 工具。Agent 可调用的外部能力，如搜索引擎、文件操作、API 调用等。通过 `BaseTool` 基类或 `@tool` 装饰器定义。

**Trigger** — 触发器。通过外部集成（webhook、定时任务等）自动触发 Crew 执行的机制。

**TypedDict** — Python 类型字典。用于定义具有固定键和类型的字典。CrewAI 的 `AgentConfig`、`TaskConfig`、`CrewMetadata` 等使用 TypedDict 定义配置结构。

**uv** — 快速 Python 包管理器。CrewAI 使用 uv 替代 pip/poetry 进行依赖管理和虚拟环境管理。

**Verbose** — 详细模式。控制 Agent 和 Crew 是否打印详细的执行日志，便于开发和调试。

**YAML** — 数据序列化格式。CrewAI 项目使用 YAML 文件（`agents.yaml`、`tasks.yaml`）配置 Agent 和 Task 的属性，支持变量占位符。
