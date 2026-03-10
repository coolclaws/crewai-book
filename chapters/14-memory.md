# 第 14 章　Memory 系统：统一记忆架构

CrewAI 的 Memory 系统经历了一次彻底的架构重构。旧版本将记忆分为 ShortTermMemory、LongTermMemory、EntityMemory、UserMemory 四种独立类型，每种各有自己的存储后端和检索逻辑。新版本将这一切统一为一个 `Memory` 类，通过 scope 层级、category 标签和 LLM 智能分析来替代硬编码的记忆类型区分。本章将深入剖析这个统一记忆架构的每一个组件。

## 14.1 整体架构概览

Memory 系统的文件结构如下：

```
memory/
├── __init__.py              # 延迟导入入口
├── unified_memory.py        # Memory 主类
├── encoding_flow.py         # 写入流水线（基于 Flow）
├── recall_flow.py           # 读取流水线（基于 Flow）
├── analyze.py               # LLM 分析模块
├── types.py                 # 核心数据类型
├── memory_scope.py          # 作用域视图
└── storage/
    ├── backend.py            # StorageBackend Protocol
    ├── lancedb_storage.py    # LanceDB 实现
    └── kickoff_task_outputs_storage.py  # Task 输出 SQLite 存储
```

与旧架构的四种记忆类型对比，新架构的核心思想是：**所有记忆都是 MemoryRecord，区别仅在于 scope 路径和 category 标签**。原来的 ShortTermMemory 对应 scope `/crew/session-xxx`，LongTermMemory 对应持久化的 scope `/crew/long-term`，EntityMemory 对应 category 包含 `entity` 的记录，UserMemory 对应 scope `/user/xxx`。统一之后，存储后端只有一个 LanceDB 表。

## 14.2 核心数据类型：MemoryRecord 与 MemoryMatch

`types.py` 定义了整个系统的基础数据结构。

### MemoryRecord：一条记忆

```python
class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(description="The textual content of the memory.")
    scope: str = Field(default="/",
        description="Hierarchical path organizing the memory (e.g. /company/team/user).")
    categories: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    embedding: list[float] | None = Field(default=None)
    source: str | None = Field(default=None)
    private: bool = Field(default=False)
```

这个设计有几个关键点：

1. **scope 是层级路径**：类似文件系统路径，如 `/company/engineering/backend`。支持前缀匹配查询，使记忆自然分层。
2. **importance 是 0-1 之间的浮点数**：影响检索时的排序权重。可以由调用者显式指定，也可以由 LLM 自动推断。
3. **source 和 private**：支持隐私控制。标记为 `private=True` 的记忆只对相同 source 的 recall 请求可见。
4. **embedding 直接存在记录里**：而非外部索引，简化了数据一致性管理。

### MemoryMatch：检索结果

```python
class MemoryMatch(BaseModel):
    record: MemoryRecord
    score: float       # 综合相关性评分
    match_reasons: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
```

`match_reasons` 说明匹配原因（semantic、recency、importance），`evidence_gaps` 记录系统查找但未找到的信息，这个设计来自 RLM（Retrieval-augmented Language Models）的启发。

### 综合评分公式

```python
def compute_composite_score(
    record: MemoryRecord,
    semantic_score: float,
    config: MemoryConfig,
) -> tuple[float, list[str]]:
    age_days = max((datetime.utcnow() - record.created_at).total_seconds() / 86400.0, 0.0)
    decay = 0.5 ** (age_days / config.recency_half_life_days)

    composite = (
        config.semantic_weight * semantic_score
        + config.recency_weight * decay
        + config.importance_weight * record.importance
    )
```

这是一个三因子加权评分：
- **语义相似度**（默认权重 0.5）：向量搜索的原始距离分数
- **时效性衰减**（默认权重 0.3）：指数衰减，半衰期默认 30 天
- **重要性**（默认权重 0.2）：记录本身的重要性标记

这个公式确保了最近的、语义相关的、重要的记忆优先浮出。

## 14.3 Memory 主类：统一记忆管理

`unified_memory.py` 中的 `Memory` 类是整个系统的门面。

```python
class Memory(BaseModel):
    llm: Annotated[BaseLLM | str, PlainValidator(_passthrough)] = Field(
        default="gpt-4o-mini")
    storage: Annotated[StorageBackend | str, PlainValidator(_passthrough)] = Field(
        default="lancedb")
    embedder: Any = Field(default=None)
    recency_weight: float = Field(default=0.3)
    semantic_weight: float = Field(default=0.5)
    importance_weight: float = Field(default=0.2)
    recency_half_life_days: int = Field(default=30)
    consolidation_threshold: float = Field(default=0.85)
    read_only: bool = Field(default=False)
```

### 延迟初始化策略

Memory 的三大依赖 -- LLM、Embedder、Storage -- 全部延迟初始化：

```python
@property
def _llm(self) -> BaseLLM:
    if self._llm_instance is None:
        from crewai.llm import LLM
        model_name = self.llm if isinstance(self.llm, str) else str(self.llm)
        self._llm_instance = LLM(model=model_name)
    return self._llm_instance

@property
def _embedder(self) -> Any:
    if self._embedder_instance is None:
        if isinstance(self.embedder, dict):
            self._embedder_instance = build_embedder(self.embedder)
        else:
            self._embedder_instance = _default_embedder()
    return self._embedder_instance
```

这种设计的好处是：`import crewai` 不会触发任何网络请求或重量级库的加载，对 Celery pre-fork 等部署模式至关重要。

### 后台写入队列

Memory 使用一个单线程的 `ThreadPoolExecutor` 做后台异步写入：

```python
_save_pool: ThreadPoolExecutor = PrivateAttr(
    default_factory=lambda: ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="memory-save"
    )
)
_pending_saves: list[Future[Any]] = PrivateAttr(default_factory=list)
_pending_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
```

`max_workers=1` 确保所有写入操作串行化，避免并发写冲突。`remember()` 方法提交到这个池并立即等待结果（同步），而 `remember_many()` 则是非阻塞的——提交后立即返回空列表。

关键设计是 **读屏障**（read barrier）：

```python
def recall(self, query, ...):
    # Read barrier: wait for any pending background saves
    self.drain_writes()
    # ... 执行搜索
```

每次 `recall()` 前先调用 `drain_writes()` 等待所有挂起的写入完成，保证读取的一致性。

## 14.4 写入流水线：EncodingFlow

`encoding_flow.py` 实现了基于 Flow 的五步写入流水线。这是 Memory 系统最精巧的部分。

### 流水线总览

```
batch_embed → intra_batch_dedup → parallel_find_similar → parallel_analyze → execute_plans
```

### Step 1: 批量 Embedding

```python
@start()
def batch_embed(self) -> None:
    items = list(self.state.items)
    texts = [item.content for item in items]
    embeddings = embed_texts(self._embedder, texts)
    for item, emb in zip(items, embeddings, strict=False):
        item.embedding = emb
```

**一次 API 调用**完成所有文本的 embedding，而非逐条调用。

### Step 2: 批内去重

```python
@listen(batch_embed)
def intra_batch_dedup(self) -> None:
    threshold = self._config.batch_dedup_threshold  # 默认 0.98
    for j in range(1, n):
        for i in range(j):
            sim = self._cosine_similarity(items[i].embedding, items[j].embedding)
            if sim >= threshold:
                items[j].dropped = True
                break
```

在同一批次内用余弦相似度矩阵检测近似重复，阈值设为 0.98（非常保守，只去除几乎完全相同的内容）。

### Step 3: 并行查找相似记录

```python
@listen(intra_batch_dedup)
def parallel_find_similar(self) -> None:
    with ThreadPoolExecutor(max_workers=min(len(active), 8)) as pool:
        futures = [(i, item, pool.submit(_search_one, item)) for i, item in active]
```

对每条未被去重的记忆，并发查询存储中是否已有相似记录。最大 8 个并发线程。

### Step 4: 并行 LLM 分析——四组分类

这一步是整个流水线的核心智能。根据两个条件将每条记忆分为四组：

| 组别 | 字段已提供? | 有相似记录? | LLM 调用次数 |
|------|-----------|-----------|------------|
| A | 是 | 否 | 0 |
| B | 是 | 是 | 1（合并决策）|
| C | 否 | 否 | 1（字段推断）|
| D | 否 | 是 | 2（推断 + 合并）|

```python
if fields_provided and not has_similar:
    # Group A: fast path, 0 LLM calls
    self._apply_defaults(item)
    item.plan = ConsolidationPlan(actions=[], insert_new=True)
elif fields_provided and has_similar:
    # Group B: consolidation only
    consol_futures[i] = pool.submit(analyze_for_consolidation, ...)
elif not fields_provided and not has_similar:
    # Group C: field resolution only
    save_futures[i] = pool.submit(analyze_for_save, ...)
else:
    # Group D: both in parallel
    save_futures[i] = pool.submit(analyze_for_save, ...)
    consol_futures[i] = pool.submit(analyze_for_consolidation, ...)
```

所有 LLM 调用在 `ThreadPoolExecutor(max_workers=10)` 中并行执行。Group A 是快速路径，完全不需要 LLM。

### 字段推断（analyze_for_save）

LLM 负责推断三个字段：

```python
class MemoryAnalysis(BaseModel):
    suggested_scope: str       # 建议的 scope 路径
    categories: list[str]      # 分类标签
    importance: float          # 重要性 0-1
    extracted_metadata: ExtractedMetadata  # 提取的实体、日期、主题
```

### 合并决策（analyze_for_consolidation）

当新记忆与已有记录高度相似（默认阈值 0.85）时，LLM 决定如何合并：

```python
class ConsolidationPlan(BaseModel):
    actions: list[ConsolidationAction]  # keep / update / delete
    insert_new: bool                     # 是否仍然插入新记录
```

每个 `ConsolidationAction` 指向一条已有记录，可以选择保留、更新其内容、或删除。

### Step 5: 执行计划

```python
@listen(parallel_analyze)
def execute_plans(self) -> None:
    with self._storage.write_lock:
        if dedup_deletes:
            self._storage.delete(record_ids=list(dedup_deletes))
        for rid, (_item_idx, new_content) in dedup_updates.items():
            # 更新已有记录
            self._storage.update(updated)
        if to_insert:
            self._storage.save(records)
```

所有存储变更在同一个写锁内原子执行：先删除、再更新、最后插入。跨 item 的操作还做了去重——如果两条新记忆都想删除同一条旧记录，只执行一次。

## 14.5 读取流水线：RecallFlow

`recall_flow.py` 实现了受 RLM 启发的智能检索流程。

### 流程图

```
analyze_query_step → filter_and_chunk → search_chunks → decide_depth
                                                            ↓
                                                    ┌─── synthesize (返回结果)
                                                    │
                                                    └─── explore_deeper → re_search → re_decide_depth
                                                                                          ↓
                                                                                   (循环回 synthesize 或继续探索)
```

### Step 1: 查询分析

```python
@start()
def analyze_query_step(self) -> QueryAnalysis:
    skip_llm = query_len < self._config.query_analysis_threshold  # 默认 250 字符
    if skip_llm:
        # 短查询：直接 embed，跳过 LLM
        analysis = QueryAnalysis(
            recall_queries=[self.state.query], complexity="simple")
    else:
        # 长查询：LLM 蒸馏为 1-3 条子查询
        analysis = analyze_query(self.state.query, available, scope_info, self._llm)
```

这是一个精巧的优化：短查询（< 250 字符）直接做向量搜索，省掉 1-3 秒的 LLM 调用。只有长查询（如完整的 task description）才通过 LLM 提炼出更精准的子查询。

LLM 的查询分析输出：

```python
class QueryAnalysis(BaseModel):
    keywords: list[str]           # 关键实体/词
    suggested_scopes: list[str]   # 建议搜索的 scope
    complexity: str               # "simple" 或 "complex"
    recall_queries: list[str]     # 1-3 条蒸馏后的搜索短语
    time_filter: str | None       # ISO 8601 时间过滤
```

### Step 2-3: Scope 筛选与并行搜索

```python
def _do_search(self) -> list[dict[str, Any]]:
    tasks = [(embedding, scope)
             for _query_text, embedding in self.state.query_embeddings
             for scope in self.state.candidate_scopes]
    with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as pool:
        # 并行搜索所有 (embedding, scope) 组合
```

如果有 3 条子查询和 5 个候选 scope，就会并行执行 15 次向量搜索。每次搜索会请求 `limit * 2` 条结果（过采样），为后续的综合评分留出余量。

### Step 4: 置信度路由

```python
@router(search_chunks)
def decide_depth(self) -> str:
    if analysis.complexity == "complex" and confidence < complex_query_threshold:
        if budget > 0: return "explore_deeper"
    if confidence >= confidence_threshold_high:
        return "synthesize"
    if budget > 0 and confidence < confidence_threshold_low:
        return "explore_deeper"
    return "synthesize"
```

这是一个 `@router` 装饰的方法，基于 Flow 的路由机制。根据置信度高低决定是直接合成结果还是继续深度探索。

### Step 5: 深度探索（可选）

```python
@listen("explore_deeper")
def recursive_exploration(self) -> list[Any]:
    self.state.exploration_budget -= 1
    for finding in self.state.chunk_findings:
        prompt = f"Query: {self.state.query}\n\nRelevant memory excerpts:\n{chunk_text}\n\n"
               + "Extract the most relevant information..."
        response = self._llm.call([{"role": "user", "content": prompt}])
        if "missing" in response.lower():
            self.state.evidence_gaps.append(response[:200])
```

将已找到的记忆片段喂给 LLM，提取更精准的信息，同时追踪 evidence gaps。探索预算（默认 1 轮）耗尽后自动停止。

### Step 6: 结果合成

去重（按 record id）、计算综合评分、排序、截取 top-k。如果存在 evidence gaps，附加到第一条结果上供调用者参考。

## 14.6 StorageBackend Protocol

`backend.py` 定义了存储后端的接口约定：

```python
@runtime_checkable
class StorageBackend(Protocol):
    def save(self, records: list[MemoryRecord]) -> None: ...
    def search(self, query_embedding: list[float],
               scope_prefix: str | None = None,
               categories: list[str] | None = None,
               limit: int = 10,
               min_score: float = 0.0) -> list[tuple[MemoryRecord, float]]: ...
    def delete(self, scope_prefix=None, categories=None,
               record_ids=None, older_than=None) -> int: ...
    def update(self, record: MemoryRecord) -> None: ...
    def get_record(self, record_id: str) -> MemoryRecord | None: ...
    def list_records(self, scope_prefix=None, limit=200, offset=0) -> list[MemoryRecord]: ...
    def get_scope_info(self, scope: str) -> ScopeInfo: ...
    def list_scopes(self, parent: str = "/") -> list[str]: ...
    def list_categories(self, scope_prefix=None) -> dict[str, int]: ...
    def count(self, scope_prefix=None) -> int: ...
    def reset(self, scope_prefix=None) -> None: ...
```

使用 Python 的 `Protocol`（结构化子类型），任何实现了这些方法签名的类都自动满足 `StorageBackend` 约束，无需继承。

## 14.7 LanceDB 存储实现

`lancedb_storage.py` 是默认的存储后端，选择 LanceDB 而非 ChromaDB 是一个重要的架构决策。

### 初始化与路径管理

```python
class LanceDBStorage:
    _path_locks: ClassVar[dict[str, threading.RLock]] = {}
    _path_locks_guard: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, path=None, table_name="memories",
                 vector_dim=None, compact_every=100):
        if path is None:
            storage_dir = os.environ.get("CREWAI_STORAGE_DIR")
            if storage_dir:
                path = Path(storage_dir) / "memory"
            else:
                path = Path(db_storage_path()) / "memory"
```

关键设计：**类级别的路径锁注册表**。当多个 Memory 实例（如 agent 级别和 crew 级别）指向同一个数据库目录时，它们共享一个 `RLock`，防止并发写冲突。锁是可重入的（`RLock`），允许 `execute_plans` 中持有外层锁时内部方法再次获取。

### 表结构

LanceDB 表中每行的字段映射：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | TEXT | UUID |
| `content` | TEXT | 记忆文本 |
| `scope` | TEXT | 层级路径 |
| `categories_str` | TEXT | JSON 序列化的分类列表 |
| `metadata_str` | TEXT | JSON 序列化的元数据 |
| `importance` | FLOAT | 重要性分数 |
| `created_at` | TEXT | ISO 8601 时间 |
| `last_accessed` | TEXT | ISO 8601 时间 |
| `source` | TEXT | 来源标识 |
| `private` | BOOL | 隐私标记 |
| `vector` | VECTOR[dim] | 嵌入向量 |

注意 `categories` 和 `metadata` 以 JSON 字符串形式存储，因为 LanceDB 的列类型不支持动态嵌套结构。

### 向量维度自动检测

```python
def save(self, records: list[MemoryRecord]) -> None:
    dim = None
    for r in records:
        if r.embedding and len(r.embedding) > 0:
            dim = len(r.embedding)
            break
    self._ensure_table(vector_dim=dim)
```

首次保存时从实际的 embedding 推断维度，然后惰性创建表。这避免了硬编码维度，支持任意 embedding 模型。

### 写入重试机制

```python
def _retry_write(self, op, *args, **kwargs):
    delay = 0.2  # seconds
    for attempt in range(6):
        try:
            return getattr(self._table, op)(*args, **kwargs)
        except OSError as e:
            if "Commit conflict" not in str(e) or attempt >= 5:
                raise
            self._table = self._db.open_table(self._table_name)  # 刷新表引用
            time.sleep(delay)
            delay *= 2
```

LanceDB 使用乐观并发控制。当两个事务重叠时，后提交的会收到 "Commit conflict" 错误。重试策略使用指数退避（0.2s、0.4s、0.8s、1.6s、3.2s），每次重试前刷新表引用以获取最新版本。

### 自动压缩

```python
self._save_count += 1
if self._compact_every > 0 and self._save_count % self._compact_every == 0:
    self._compact_async()
```

每 100 次 `save()` 调用后在后台线程执行 `table.optimize()`，合并碎片文件。同时在打开已有表时也会触发一次后台压缩，确保历史碎片不影响查询性能。

## 14.8 MemoryScope 与 MemorySlice

`memory_scope.py` 提供了两种记忆视图，用于限制操作的范围。

### MemoryScope：单一路径视图

```python
class MemoryScope(BaseModel):
    root_path: str = Field(default="/")

    def remember(self, content, scope="/", ...):
        path = self._scope_path(scope)  # 相对路径拼接到 root_path
        return self._memory.remember(content, scope=path, ...)

    def subscope(self, path: str) -> MemoryScope:
        new_root = f"{base}/{child}"
        return MemoryScope(memory=self._memory, root_path=new_root)
```

`MemoryScope` 把所有操作限制在一个 root_path 下。`subscope()` 可以创建更窄的嵌套视图，形成层级结构。

### MemorySlice：多路径视图

```python
class MemorySlice(BaseModel):
    scopes: list[str]
    categories: list[str] | None = None
    read_only: bool = True

    def recall(self, query, ...):
        all_matches = []
        for sc in self.scopes:
            matches = self._memory.recall(query, scope=sc, ...)
            all_matches.extend(matches)
        # 去重、排序、截取 top-k
```

`MemorySlice` 跨多个 scope 搜索，结果合并后重新排序。默认只读，适合 Agent 在执行时查看多个相关 scope 的记忆但不写入。

## 14.9 旧架构与新架构的映射

理解旧的四种记忆类型如何映射到新架构有助于理解设计动机：

| 旧类型 | 新架构等价物 |
|--------|------------|
| ShortTermMemory | scope `/crew/{crew_id}/session/{session_id}`, 单次 kickoff 内的记忆 |
| LongTermMemory | scope `/crew/{crew_id}/long-term`, 跨 kickoff 持久化，由 KickoffTaskOutputsSQLiteStorage 辅助 |
| EntityMemory | category 包含 `entity`，metadata 中有 `entities` 字段（由 LLM 自动提取） |
| UserMemory / ExternalMemory | scope `/user/{user_id}`, source 字段标记来源, private=True 控制可见性 |

`KickoffTaskOutputsSQLiteStorage` 仍然保留，但它不是记忆系统的一部分，而是用于 Task 输出的持久化存储，支持 replay 功能。其 SQLite 表结构为：

```sql
CREATE TABLE latest_kickoff_task_outputs (
    task_id TEXT PRIMARY KEY,
    expected_output TEXT,
    output JSON,
    task_index INTEGER,
    inputs JSON,
    was_replayed BOOLEAN,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

## 14.10 LLM 分析模块

`analyze.py` 集中管理所有 LLM 调用的 prompt 和响应解析。

### 记忆提取

```python
def extract_memories_from_content(content: str, llm: Any) -> list[str]:
    user = _get_prompt("extract_memories_user").format(content=content)
    messages = [
        {"role": "system", "content": _get_prompt("extract_memories_system")},
        {"role": "user", "content": user},
    ]
    if getattr(llm, "supports_function_calling", lambda: False)():
        response = llm.call(messages, response_model=ExtractedMemories)
    else:
        response = llm.call(messages)
        data = json.loads(response)
        return ExtractedMemories.model_validate(data).memories
```

支持两条路径：function calling（结构化输出）和纯文本 JSON 解析。失败时回退到将整个内容作为单条记忆保存，确保不丢失数据。

### 结构化输出模式

所有 LLM 分析都使用 Pydantic 模型作为 `response_model`：
- `ExtractedMemories`：从原始内容提取离散记忆
- `MemoryAnalysis`：推断 scope、categories、importance
- `QueryAnalysis`：分析查询并蒸馏子查询
- `ConsolidationPlan`：合并决策

`ExtractedMetadata` 使用 `extra="forbid"` 配置，因为 OpenAI 的 function calling 要求 `additionalProperties: false`：

```python
class ExtractedMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entities: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
```

## 14.11 事件总线集成

Memory 的每个关键操作都通过 `crewai_event_bus` 发出事件：

- `MemorySaveStartedEvent` / `MemorySaveCompletedEvent` / `MemorySaveFailedEvent`
- `MemoryQueryStartedEvent` / `MemoryQueryCompletedEvent` / `MemoryQueryFailedEvent`

事件中包含 `source_type="unified_memory"` 标识来源，`save_time_ms` 和 `query_time_ms` 记录耗时。后台写入的事件在后台线程中发出，并用 try/except 包裹以处理进程退出时事件总线已关闭的情况。

## 14.12 延迟导入机制

`__init__.py` 使用 `__getattr__` 实现模块级延迟导入：

```python
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Memory": ("crewai.memory.unified_memory", "Memory"),
    "EncodingFlow": ("crewai.memory.encoding_flow", "EncodingFlow"),
}

def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val  # 缓存到模块全局，只导入一次
        return val
    raise AttributeError(...)
```

这确保 `import crewai` 不会触发 lancedb 的加载（lancedb 依赖 pyarrow 等重量级库），只有实际使用 Memory 时才导入。

## 本章要点

- CrewAI 新版 Memory 系统**统一了四种旧记忆类型**为单一 `Memory` 类，通过 scope 层级路径和 category 标签替代硬编码分类
- **MemoryRecord** 是核心数据单元，包含 content、scope、categories、importance、embedding、source、private 等字段
- **EncodingFlow** 实现五步写入流水线：批量 Embedding -> 批内去重 -> 并行查找相似 -> 并行 LLM 分析（四组分类优化）-> 原子执行计划
- **RecallFlow** 实现 RLM 启发的智能检索：查询分析 -> Scope 筛选 -> 并行搜索 -> 置信度路由 -> 可选深度探索 -> 结果合成
- **综合评分公式**融合语义相似度（0.5）、时效性衰减（0.3）和重要性（0.2）三个因子
- **LanceDB** 作为默认存储后端，支持乐观并发控制、自动压缩、向量维度自动检测
- **StorageBackend Protocol** 允许自定义存储实现，无需继承基类
- **MemoryScope** 和 **MemorySlice** 提供作用域视图和多路径切片视图
- 后台写入队列 + 读屏障确保**写入不阻塞 Agent 执行**，同时保证**读取一致性**
- 所有重量级依赖**延迟加载**，避免 import 时的副作用
