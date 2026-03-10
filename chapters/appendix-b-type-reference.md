# 附录 B：关键类型速查

本附录列出 CrewAI 中最重要的类及其源码位置、关键字段和用途，供开发过程中快速查阅。

## B.1 核心模型类

| 类名 | 源码路径 | 基类 | 关键字段 | 说明 |
|------|----------|------|----------|------|
| `Agent` | `agent.py` | `BaseAgent` | `role`, `goal`, `backstory`, `llm`, `tools`, `memory`, `verbose`, `allow_delegation`, `max_iter`, `max_retry_limit` | 核心 Agent 类，封装 LLM 交互和工具调用 |
| `BaseAgent` | `agents/agent_builder/base_agent.py` | `BaseModel` | `id`, `role`, `goal`, `backstory`, `cache`, `verbose`, `max_rpm`, `tools` | Agent 抽象基类 |
| `Task` | `task.py` | `BaseModel` | `name`, `description`, `expected_output`, `agent`, `tools`, `context`, `output_json`, `output_pydantic`, `output_file`, `async_execution`, `human_input`, `guardrail` | 任务定义，包含描述、期望输出和执行配置 |
| `Crew` | `crew.py` | `BaseModel` | `agents`, `tasks`, `process`, `verbose`, `memory`, `embedder`, `knowledge`, `planning`, `before_kickoff_callbacks`, `after_kickoff_callbacks` | Crew 编排容器，管理 Agent 和 Task 的协作 |
| `Process` | `process.py` | `str, Enum` | `sequential`, `hierarchical` | 执行流程枚举 |

## B.2 LLM 相关

| 类名 | 源码路径 | 基类 | 关键字段 | 说明 |
|------|----------|------|----------|------|
| `LLM` | `llm.py` | `BaseLLM` | `model`, `temperature`, `max_tokens`, `timeout`, `api_key`, `base_url`, `top_p`, `stop`, `response_format` | 统一 LLM 接口，基于 litellm |
| `BaseLLM` | `llms/base_llm.py` | `BaseModel` | `model`, `temperature`, `max_tokens`, `callbacks` | LLM 抽象基类 |

## B.3 Tool 系统

| 类名 | 源码路径 | 基类 | 关键字段 | 说明 |
|------|----------|------|----------|------|
| `BaseTool` | `tools/base_tool.py` | `BaseModel` | `name`, `description`, `args_schema`, `cache_function` | 工具抽象基类 |
| `StructuredTool` | `tools/structured_tool.py` | `BaseTool` | `func`, `args_schema` | 从函数创建的结构化工具 |
| `tool` (装饰器) | `tools/tool.py` | — | — | 将函数转换为 Tool 的装饰器 |
| `ToolResult` | `tools/base_tool.py` | — | `result`, `result_as_answer` | 工具执行结果 |

## B.4 Flow 系统

| 类名 | 源码路径 | 基类 | 关键字段 | 说明 |
|------|----------|------|----------|------|
| `Flow` | `flow/flow.py` | `Generic[T]` | `initial_state`, `_methods`, `_listeners`, `_routers`, `_start_methods` | Flow 基类，管理有状态的工作流 |
| `FlowState` | `flow/flow.py` | `BaseModel` | （用户自定义） | Flow 状态基类 |
| `@start` | `flow/flow.py` | — | — | 标记 Flow 入口方法 |
| `@listen` | `flow/flow.py` | — | — | 标记监听上游方法的方法 |
| `@router` | `flow/flow.py` | — | — | 标记条件路由方法 |

## B.5 Memory 系统

| 类名 | 源码路径 | 基类 | 关键字段 | 说明 |
|------|----------|------|----------|------|
| `Memory` | `memory/memory.py` | `BaseModel` | `storage`, `embedder_config` | Memory 统一接口 |
| `ShortTermMemory` | `memory/short_term/short_term_memory.py` | `Memory` | `storage` | 短期记忆（单次运行） |
| `LongTermMemory` | `memory/long_term/long_term_memory.py` | `Memory` | `storage` | 长期记忆（跨运行持久化） |
| `EntityMemory` | `memory/entity/entity_memory.py` | `Memory` | `storage` | 实体记忆 |
| `UserMemory` | `memory/user/user_memory.py` | `Memory` | `storage` | 用户级记忆 |
| `ExternalMemory` | `memory/external/external_memory.py` | `Memory` | `storage` | 外部存储记忆 |

## B.6 Knowledge 系统

| 类名 | 源码路径 | 基类 | 关键字段 | 说明 |
|------|----------|------|----------|------|
| `Knowledge` | `knowledge/knowledge.py` | `BaseModel` | `sources`, `embedder_config`, `storage`, `collection_name` | Knowledge 管理器 |
| `BaseKnowledgeSource` | `knowledge/source/base_knowledge_source.py` | `BaseModel` | `content`, `metadata`, `chunk_size`, `chunk_overlap` | 知识源基类 |
| `StringKnowledgeSource` | `knowledge/source/string_knowledge_source.py` | `BaseKnowledgeSource` | `content` | 字符串知识源 |
| `FileKnowledgeSource` | `knowledge/source/file_knowledge_source.py` | `BaseKnowledgeSource` | `file_paths` | 文件知识源 |

## B.7 Storage 层

| 类名 | 源码路径 | 基类 | 关键字段 | 说明 |
|------|----------|------|----------|------|
| `LanceDBStorage` | `memory/storage/lance_storage.py` | — | `db_path`, `table_name`, `embedder_config` | LanceDB 向量存储 |
| `RAGStorage` | `memory/storage/rag_storage.py` | — | `type`, `embedder_config` | RAG 存储适配器 |
| `LTMSQLiteStorage` | `memory/storage/ltm_sqlite_storage.py` | — | `db_path` | 长期记忆 SQLite 存储 |
| `KickoffTaskOutputsSQLiteStorage` | `memory/storage/kickoff_task_outputs_storage.py` | — | `db_path` | Task 输出存储（供 replay 使用） |

## B.8 Pipeline 系统

| 类名 | 源码路径 | 基类 | 关键字段 | 说明 |
|------|----------|------|----------|------|
| `Pipeline` | `pipeline/pipeline.py` | `BaseModel` | `stages`, `name` | Pipeline 容器 |
| `PipelineKickoffResult` | `pipeline/pipeline_output.py` | — | `raw`, `json_dict`, `pydantic`, `token_usage` | Pipeline 执行结果 |

## B.9 Project 模板

| 类名/装饰器 | 源码路径 | 说明 |
|-------------|----------|------|
| `CrewBase` | `project/crew_base.py` | 类装饰器，将类转为声明式 Crew 定义 |
| `CrewBaseMeta` | `project/crew_base.py` | 元类，注入配置加载和变量映射 |
| `@agent` | `project/annotations.py` | 标记方法为 Agent 工厂 |
| `@task` | `project/annotations.py` | 标记方法为 Task 工厂 |
| `@crew` | `project/annotations.py` | 标记方法为 Crew 组装入口 |
| `@before_kickoff` | `project/annotations.py` | 标记 kickoff 前回调 |
| `@after_kickoff` | `project/annotations.py` | 标记 kickoff 后回调 |
| `@llm` | `project/annotations.py` | 标记方法为 LLM 工厂 |
| `@tool` | `project/annotations.py` | 标记方法为 Tool 工厂 |
| `@callback` | `project/annotations.py` | 标记方法为回调工厂 |
| `@cache_handler` | `project/annotations.py` | 标记方法为缓存处理器工厂 |
| `@output_json` | `project/annotations.py` | 标记类为 JSON 输出格式 |
| `@output_pydantic` | `project/annotations.py` | 标记类为 Pydantic 输出格式 |

## B.10 事件系统

| 类名 | 源码路径 | 基类 | 关键字段 | 说明 |
|------|----------|------|----------|------|
| `CrewEvent` | `events/base_events.py` | — | `type`, `timestamp`, `data` | 事件基类 |
| `EventEmitter` | `events/event_emitter.py` | — | `_listeners` | 事件发射器 |
| `EventListener` | `events/event_listener.py` | — | `_handlers` | 事件监听器 |

## B.11 输出类型

| 类名 | 源码路径 | 关键字段 | 说明 |
|------|----------|----------|------|
| `TaskOutput` | `tasks/task_output.py` | `description`, `raw`, `json_dict`, `pydantic`, `agent`, `output_format` | 单个 Task 的输出 |
| `CrewOutput` | `crews/crew_output.py` | `raw`, `json_dict`, `pydantic`, `tasks_output`, `token_usage` | Crew 执行的整体输出 |

## B.12 配置类型

| 类名 | 源码路径 | 说明 |
|------|----------|------|
| `AgentConfig` | `project/crew_base.py` | Agent YAML 配置的 TypedDict |
| `TaskConfig` | `project/crew_base.py` | Task YAML 配置的 TypedDict |
| `CrewMetadata` | `project/wrappers.py` | Crew 类元数据的 TypedDict |
| `CrewInstance` | `project/wrappers.py` | Crew 实例的 Protocol 定义 |
| `CrewClass` | `project/wrappers.py` | Crew 类的 Protocol 定义 |

## B.13 Wrapper 类型

| 类名 | 源码路径 | 标记属性 | 说明 |
|------|----------|----------|------|
| `DecoratedMethod` | `project/wrappers.py` | — | 装饰器方法的基类 |
| `AgentMethod` | `project/wrappers.py` | `is_agent` | @agent 方法包装器 |
| `TaskMethod` | `project/wrappers.py` | `is_task` | @task 方法包装器 |
| `LLMMethod` | `project/wrappers.py` | `is_llm` | @llm 方法包装器 |
| `ToolMethod` | `project/wrappers.py` | `is_tool` | @tool 方法包装器 |
| `CallbackMethod` | `project/wrappers.py` | `is_callback` | @callback 方法包装器 |
| `CacheHandlerMethod` | `project/wrappers.py` | `is_cache_handler` | @cache_handler 方法包装器 |
| `BeforeKickoffMethod` | `project/wrappers.py` | `is_before_kickoff` | @before_kickoff 方法包装器 |
| `AfterKickoffMethod` | `project/wrappers.py` | `is_after_kickoff` | @after_kickoff 方法包装器 |
| `OutputJsonClass` | `project/wrappers.py` | `is_output_json` | @output_json 类包装器 |
| `OutputPydanticClass` | `project/wrappers.py` | `is_output_pydantic` | @output_pydantic 类包装器 |
| `BoundTaskMethod` | `project/wrappers.py` | `is_task` | 绑定到实例的 Task 方法 |

> **注意**：所有源码路径均相对于 `crewai/` 根目录。实际文件系统中的完整路径为 `src/crewai/<path>`。
