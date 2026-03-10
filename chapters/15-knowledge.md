# 第 15 章　Knowledge：文档摄入与 RAG

如果说 Memory 系统是 Agent 的"经验记忆"，那么 Knowledge 系统就是 Agent 的"知识库"。Memory 存储的是执行过程中产生的交互记忆，而 Knowledge 存储的是预先准备的文档知识——PDF 手册、CSV 数据表、JSON 配置、纯文本文件等。本章深入分析 Knowledge 系统的文档摄入管道、分块策略、向量存储，以及与 Agent 查询的集成。

## 15.1 整体架构

Knowledge 系统的文件结构：

```
knowledge/
├── __init__.py
├── knowledge.py              # Knowledge 主类
├── knowledge_config.py       # 查询配置
├── source/
│   ├── base_knowledge_source.py      # 抽象基类
│   ├── base_file_knowledge_source.py # 文件类基类
│   ├── pdf_knowledge_source.py       # PDF 源
│   ├── csv_knowledge_source.py       # CSV 源
│   ├── excel_knowledge_source.py     # Excel 源
│   ├── json_knowledge_source.py      # JSON 源
│   ├── text_file_knowledge_source.py # 纯文本源
│   ├── string_knowledge_source.py    # 字符串源
│   ├── crew_docling_source.py        # Docling 多格式源
│   └── utils/
│       └── source_helper.py          # 文件类型自动路由
├── storage/
│   ├── base_knowledge_storage.py     # 存储抽象接口
│   └── knowledge_storage.py          # ChromaDB 实现
└── utils/
    └── knowledge_utils.py            # 上下文提取工具
```

数据流概览：

```
原始文档 → KnowledgeSource.load_content()
         → KnowledgeSource._chunk_text()
         → KnowledgeSource._save_documents()
         → KnowledgeStorage.save(chunks)
         → ChromaDB / RAG Client（embedding + 持久化）

查询 → Knowledge.query(query_strings)
     → KnowledgeStorage.search(query, limit, threshold)
     → ChromaDB 向量搜索
     → list[SearchResult]
```

与 Memory 系统不同，Knowledge 使用 **ChromaDB**（通过 RAG 抽象层）而非 LanceDB。这是因为 Knowledge 的需求更简单：写入一次、多次查询，不需要 scope 层级、合并决策、重要性评分等复杂逻辑。

## 15.2 Knowledge 主类

`knowledge.py` 中的 `Knowledge` 类是整个系统的入口：

```python
class Knowledge(BaseModel):
    sources: list[BaseKnowledgeSource] = Field(default_factory=list)
    storage: KnowledgeStorage | None = Field(default=None)
    embedder: EmbedderConfig | None = None
    collection_name: str | None = None
```

### 初始化

```python
def __init__(self, collection_name: str, sources: list[BaseKnowledgeSource],
             embedder=None, storage=None, **data):
    super().__init__(**data)
    if storage:
        self.storage = storage
    else:
        self.storage = KnowledgeStorage(
            embedder=embedder, collection_name=collection_name)
    self.sources = sources
```

`collection_name` 是必需参数，它决定了 ChromaDB 中的 collection 名称。如果不传 `storage`，会自动创建一个 `KnowledgeStorage` 实例。`embedder` 参数允许自定义 embedding 模型。

### 文档摄入

```python
def add_sources(self) -> None:
    for source in self.sources:
        source.storage = self.storage
        source.add()
```

遍历所有 source，将 storage 引用注入到每个 source 中，然后调用 `source.add()` 触发摄入流程。每个 source 负责自己的文件读取、内容转换、分块，最后通过 `storage.save()` 持久化。

### 查询

```python
def query(self, query: list[str], results_limit: int = 5,
          score_threshold: float = 0.6) -> list[SearchResult]:
    return self.storage.search(
        query, limit=results_limit, score_threshold=score_threshold)
```

查询接口接受一个字符串列表（会被合并为单一查询文本），返回 `SearchResult` 列表。`score_threshold` 默认 0.6，过滤掉相关性过低的结果。

### KnowledgeConfig

```python
class KnowledgeConfig(BaseModel):
    results_limit: int = Field(default=5)
    score_threshold: float = Field(default=0.6)
```

简单的配置类，供 Crew 或 Task 级别引用。

## 15.3 BaseKnowledgeSource：文档源抽象

`base_knowledge_source.py` 定义了所有知识源的抽象基类：

```python
class BaseKnowledgeSource(BaseModel, ABC):
    chunk_size: int = 4000
    chunk_overlap: int = 200
    chunks: list[str] = Field(default_factory=list)
    chunk_embeddings: list[np.ndarray] = Field(default_factory=list)
    storage: KnowledgeStorage | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
    collection_name: str | None = Field(default=None)

    @abstractmethod
    def validate_content(self) -> Any: ...

    @abstractmethod
    def add(self) -> None: ...

    @abstractmethod
    async def aadd(self) -> None: ...
```

### 分块策略

```python
def _chunk_text(self, text: str) -> list[str]:
    return [
        text[i : i + self.chunk_size]
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
    ]
```

这是一个简单的**固定窗口滑动分块**策略：
- `chunk_size=4000`：每个块 4000 字符
- `chunk_overlap=200`：相邻块重叠 200 字符

滑动步长为 `chunk_size - chunk_overlap = 3800`。重叠确保不会在句子或段落边界丢失上下文。这种策略虽然简单，但对大多数文档效果不错。

### 保存到存储

```python
def _save_documents(self) -> None:
    if self.storage:
        self.storage.save(self.chunks)
    else:
        raise ValueError("No storage found to save documents.")
```

分块后直接将字符串列表传给 storage。注意这里传的是纯文本块，**embedding 在 storage 层计算**，而非在 source 层。

## 15.4 BaseFileKnowledgeSource：文件类基类

`base_file_knowledge_source.py` 为所有文件类型的知识源提供通用的文件路径处理逻辑：

```python
class BaseFileKnowledgeSource(BaseKnowledgeSource, ABC):
    file_path: Path | list[Path] | str | list[str] | None = Field(default=None)
    file_paths: Path | list[Path] | str | list[str] | None = Field(default_factory=list)
    content: dict[Path, str] = Field(init=False, default_factory=dict)
    safe_file_paths: list[Path] = Field(default_factory=list)
```

### 初始化流程

```python
def model_post_init(self, _: Any) -> None:
    self.safe_file_paths = self._process_file_paths()
    self.validate_content()
    self.content = self.load_content()
```

三步：处理路径 -> 验证文件存在 -> 加载内容。`load_content()` 是抽象方法，由各子类实现。

### 路径解析

```python
def convert_to_path(self, path: Path | str) -> Path:
    return Path(KNOWLEDGE_DIRECTORY + "/" + path) if isinstance(path, str) else path
```

字符串路径会自动拼接 `KNOWLEDGE_DIRECTORY` 前缀（通常是项目根目录下的 `knowledge/` 目录）。`Path` 类型则保持原样，支持绝对路径。

`file_path` 是已废弃的旧接口（会打印 warning），新代码应使用 `file_paths`（支持单个或列表）。

## 15.5 七种内置 Knowledge Source

### PDFKnowledgeSource

```python
class PDFKnowledgeSource(BaseFileKnowledgeSource):
    def load_content(self) -> dict[Path, str]:
        pdfplumber = self._import_pdfplumber()
        content = {}
        for path in self.safe_file_paths:
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            content[path] = text
        return content
```

使用 `pdfplumber` 库逐页提取文本。依赖是动态导入的——如果未安装，会抛出清晰的 `ImportError` 提示用户安装。

`add()` 方法遍历每个文件的内容，分块后保存：

```python
def add(self) -> None:
    for text in self.content.values():
        new_chunks = self._chunk_text(text)
        self.chunks.extend(new_chunks)
    self._save_documents()
```

### CSVKnowledgeSource

```python
class CSVKnowledgeSource(BaseFileKnowledgeSource):
    def load_content(self) -> dict[Path, str]:
        content_dict = {}
        for file_path in self.safe_file_paths:
            with open(file_path, "r", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                content = ""
                for row in reader:
                    content += " ".join(row) + "\n"
                content_dict[file_path] = content
        return content_dict
```

将 CSV 的每行转为空格分隔的文本，行间用换行符分隔。这种扁平化处理使得 CSV 数据可以被正常的文本分块和 embedding 处理。

### ExcelKnowledgeSource

```python
class ExcelKnowledgeSource(BaseKnowledgeSource):
    content: dict[Path, dict[str, str]] = Field(default_factory=dict)

    def _load_content(self) -> dict[Path, dict[str, str]]:
        pd = self._import_dependencies()  # 动态导入 pandas
        for file_path in self.safe_file_paths:
            with pd.ExcelFile(file_path) as xl:
                sheet_dict = {
                    str(sheet_name): str(pd.read_excel(xl, sheet_name).to_csv(index=False))
                    for sheet_name in xl.sheet_names
                }
            content_dict[file_path] = sheet_dict
        return content_dict
```

Excel 源的特殊之处：
1. 它**不继承 BaseFileKnowledgeSource**，而是直接继承 `BaseKnowledgeSource`（因为 content 的类型不同，是嵌套字典）
2. 每个 sheet 独立读取，转为 CSV 格式文本
3. 依赖 `pandas`，同样是动态导入

### JSONKnowledgeSource

```python
class JSONKnowledgeSource(BaseFileKnowledgeSource):
    def load_content(self) -> dict[Path, str]:
        for path in self.safe_file_paths:
            with open(path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
            content[path] = self._json_to_text(data)
        return content

    def _json_to_text(self, data: Any, level: int = 0) -> str:
        indent = "  " * level
        if isinstance(data, dict):
            for key, value in data.items():
                text += f"{indent}{key}: {self._json_to_text(value, level + 1)}\n"
        elif isinstance(data, list):
            for item in data:
                text += f"{indent}- {self._json_to_text(item, level + 1)}\n"
        else:
            text += f"{data!s}"
        return text
```

JSON 到文本的转换是递归的，保留层级结构（用缩进表示），使嵌套 JSON 在向量搜索时仍然保持可读的语义信息。

### TextFileKnowledgeSource

```python
class TextFileKnowledgeSource(BaseFileKnowledgeSource):
    def load_content(self) -> dict[Path, str]:
        for path in self.safe_file_paths:
            with open(path, "r", encoding="utf-8") as f:
                content[path] = f.read()
        return content
```

最简单的源——直接读取文件内容。

### StringKnowledgeSource

```python
class StringKnowledgeSource(BaseKnowledgeSource):
    content: str = Field(...)

    def add(self) -> None:
        new_chunks = self._chunk_text(self.content)
        self.chunks.extend(new_chunks)
        self._save_documents()
```

唯一不涉及文件的源，直接接受字符串输入。适合动态生成的内容或测试场景。

### CrewDoclingSource

```python
class CrewDoclingSource(BaseKnowledgeSource):
    file_paths: list[Path | str] = Field(default_factory=list)
    content: list[DoclingDocument] = Field(default_factory=list)
    document_converter: DocumentConverter = Field(
        default_factory=lambda: DocumentConverter(
            allowed_formats=[
                InputFormat.MD, InputFormat.ASCIIDOC, InputFormat.PDF,
                InputFormat.DOCX, InputFormat.HTML, InputFormat.IMAGE,
                InputFormat.XLSX, InputFormat.PPTX,
            ]
        )
    )
```

`CrewDoclingSource` 是最强大的知识源，基于 IBM 的 `docling` 库。它的关键优势：

1. **统一处理多种格式**：PDF、DOCX、HTML、Markdown、AsciiDoc、Image、XLSX、PPTX
2. **结构化分块**：使用 `HierarchicalChunker` 而非简单的滑动窗口

```python
def _chunk_doc(self, doc: DoclingDocument) -> Iterator[str]:
    chunker = HierarchicalChunker()
    for chunk in chunker.chunk(doc):
        yield chunk.text
```

`HierarchicalChunker` 理解文档的结构（标题、段落、列表），在自然边界处分块，比固定窗口分块质量更高。

3. **URL 支持**：除了本地文件，还支持 HTTP/HTTPS URL

```python
def validate_content(self) -> list[Path | str]:
    for path in self.file_paths:
        if isinstance(path, str):
            if path.startswith(("http://", "https://")):
                if self._validate_url(path):
                    processed_paths.append(path)
            else:
                local_path = Path(KNOWLEDGE_DIRECTORY + "/" + path)
                processed_paths.append(local_path)
```

## 15.6 类层次结构

```
BaseKnowledgeSource (ABC)
├── BaseFileKnowledgeSource (ABC)
│   ├── PDFKnowledgeSource
│   ├── CSVKnowledgeSource
│   ├── JSONKnowledgeSource
│   └── TextFileKnowledgeSource
├── StringKnowledgeSource
├── ExcelKnowledgeSource      # 注意：不继承 BaseFileKnowledgeSource
└── CrewDoclingSource
```

`ExcelKnowledgeSource` 没有继承 `BaseFileKnowledgeSource` 是一个值得注意的设计选择。原因是 Excel 的 `content` 类型是 `dict[Path, dict[str, str]]`（文件 -> sheet -> CSV 文本），而 `BaseFileKnowledgeSource` 的 `content` 类型是 `dict[Path, str]`。为避免类型冲突，Excel 源直接继承 `BaseKnowledgeSource` 并自行实现路径处理逻辑（有一些代码重复）。

## 15.7 SourceHelper：文件类型自动路由

```python
class SourceHelper:
    SUPPORTED_FILE_TYPES: ClassVar[list[str]] = [
        ".csv", ".pdf", ".json", ".txt", ".xlsx", ".xls",
    ]
    _FILE_TYPE_MAP: ClassVar[dict[str, type[BaseKnowledgeSource]]] = {
        ".csv": CSVKnowledgeSource,
        ".pdf": PDFKnowledgeSource,
        ".json": JSONKnowledgeSource,
        ".txt": TextFileKnowledgeSource,
        ".xlsx": ExcelKnowledgeSource,
        ".xls": ExcelKnowledgeSource,
    }

    @classmethod
    def get_source(cls, file_path: str, metadata=None) -> BaseKnowledgeSource:
        lower_path = file_path.lower()
        for ext, source_cls in cls._FILE_TYPE_MAP.items():
            if lower_path.endswith(ext):
                return source_cls(file_path=[file_path], metadata=metadata)
        raise ValueError(f"Unsupported file type: {file_path}")
```

`SourceHelper` 根据文件扩展名自动选择正确的 `KnowledgeSource` 类。注意 `.xls` 和 `.xlsx` 都映射到 `ExcelKnowledgeSource`。

## 15.8 KnowledgeStorage：向量存储层

### BaseKnowledgeStorage 接口

```python
class BaseKnowledgeStorage(ABC):
    @abstractmethod
    def search(self, query: list[str], limit=5,
               metadata_filter=None, score_threshold=0.6) -> list[SearchResult]: ...
    @abstractmethod
    def save(self, documents: list[str]) -> None: ...
    @abstractmethod
    def reset(self) -> None: ...
    # + async 变体
```

接口非常精简：save、search、reset，加上对应的 async 版本。

### KnowledgeStorage 实现

```python
class KnowledgeStorage(BaseKnowledgeStorage):
    def __init__(self, embedder=None, collection_name=None):
        self.collection_name = collection_name
        self._client: BaseClient | None = None

        if embedder:
            embedding_function = build_embedder(embedder)
            config = ChromaDBConfig(
                embedding_function=cast(ChromaEmbeddingFunctionWrapper, embedding_function))
            self._client = create_client(config)
```

`KnowledgeStorage` 通过 CrewAI 的 RAG 抽象层与 ChromaDB 交互。如果提供了自定义 embedder，就用它创建一个专用的 client；否则使用全局默认 client。

### 搜索流程

```python
def search(self, query: list[str], limit=5, metadata_filter=None,
           score_threshold=0.6) -> list[SearchResult]:
    client = self._get_client()
    collection_name = (
        f"knowledge_{self.collection_name}" if self.collection_name else "knowledge")
    query_text = " ".join(query) if len(query) > 1 else query[0]
    return client.search(
        collection_name=collection_name,
        query=query_text,
        limit=limit,
        metadata_filter=metadata_filter,
        score_threshold=score_threshold,
    )
```

关键细节：
1. **collection 命名约定**：`knowledge_{collection_name}`，前缀 `knowledge_` 避免与其他用途的 collection 冲突
2. **多查询合并**：如果传入多个查询字符串，用空格拼接为一个查询
3. **score_threshold**：低于此阈值的结果会被过滤掉

### 保存流程

```python
def save(self, documents: list[str]) -> None:
    client = self._get_client()
    collection_name = f"knowledge_{self.collection_name}" if self.collection_name else "knowledge"
    client.get_or_create_collection(collection_name=collection_name)
    rag_documents: list[BaseRecord] = [{"content": doc} for doc in documents]
    client.add_documents(collection_name=collection_name, documents=rag_documents)
```

每个文本块被包装为 `{"content": doc}` 格式的 `BaseRecord`。`get_or_create_collection` 确保 collection 存在。embedding 的计算在 `client.add_documents` 内部完成。

### Embedding 维度不匹配处理

```python
except Exception as e:
    if "dimension mismatch" in str(e).lower():
        raise ValueError(
            "Embedding dimension mismatch. Make sure you're using the same embedding model "
            "across all operations with this collection."
            "Try resetting the collection using `crewai reset-memories -a`"
        ) from e
```

这是一个常见的运行时错误：当用户更换了 embedding 模型但没有重置 collection 时，新旧 embedding 维度不一致。框架提供了清晰的错误提示和解决方案。

## 15.9 Knowledge 与 Memory 的架构对比

| 维度 | Knowledge | Memory |
|------|-----------|--------|
| 用途 | 预先准备的文档知识 | 运行时产生的经验记忆 |
| 写入模式 | 批量导入，一次写入 | 持续增量写入 |
| 向量数据库 | ChromaDB（通过 RAG 抽象层）| LanceDB（直接集成）|
| 分块策略 | Source 层分块 | Storage 层不分块（单条记忆即一条记录）|
| LLM 参与 | 无（纯向量检索）| 深度参与（分析、合并、查询蒸馏）|
| Scope/层级 | 无（flat collection）| 层级路径 |
| 合并策略 | 无（追加写入）| LLM 驱动的 consolidation |
| 评分 | 纯语义相似度 + 阈值 | 三因子综合评分 |

这两个系统互补但独立：Knowledge 提供静态的"教科书知识"，Memory 提供动态的"工作记忆"。

## 15.10 数据流全景

让我们追踪一个完整的 PDF 文档从摄入到查询的全过程：

**阶段一：摄入**

```
1. Knowledge(collection_name="manual", sources=[PDFKnowledgeSource(file_paths=["guide.pdf"])])
2. knowledge.add_sources()
   ↓
3. PDFKnowledgeSource.add()
   - pdfplumber 逐页提取文本
   - _chunk_text() 按 4000 字符窗口、200 字符重叠分块
   - 假设 PDF 有 50 页，产生约 20 个 chunk
   ↓
4. _save_documents() → storage.save(chunks)
   ↓
5. KnowledgeStorage.save()
   - client.get_or_create_collection("knowledge_manual")
   - client.add_documents(20 个 BaseRecord)
   - ChromaDB 内部计算 20 个 embedding 并持久化
```

**阶段二：查询**

```
1. knowledge.query(["How to configure logging?"])
   ↓
2. KnowledgeStorage.search()
   - client.search(collection_name="knowledge_manual", query="How to configure logging?")
   - ChromaDB 计算 query embedding，执行 ANN 搜索
   - 返回 top-5 结果，过滤 score < 0.6
   ↓
3. list[SearchResult] 返回给调用者
```

**阶段三：Agent 集成**

在 Agent 执行 Task 时，Knowledge 查询结果被注入到 prompt 中：

```python
def extract_knowledge_context(knowledge_snippets: list[SearchResult]) -> str:
    valid_snippets = [
        result["content"]
        for result in knowledge_snippets
        if result and result.get("content")
    ]
    snippet = "\n".join(valid_snippets)
    return f"Additional Information: {snippet}" if valid_snippets else ""
```

`knowledge_utils.py` 中的 `extract_knowledge_context` 函数将搜索结果格式化为 `"Additional Information: ..."` 前缀的上下文字符串，附加到 Agent 的 prompt 中。

## 15.11 异步支持

Knowledge 系统全面支持 async 操作：

```python
# Knowledge 主类
async def aquery(self, query, results_limit=5, score_threshold=0.6):
    return await self.storage.asearch(query, limit=results_limit, score_threshold=score_threshold)

async def aadd_sources(self):
    for source in self.sources:
        source.storage = self.storage
        await source.aadd()

# 每个 Source
async def aadd(self):
    # 与 add() 相同的逻辑
    await self._asave_documents()

# Storage
async def asave(self, documents):
    await client.aadd_documents(collection_name=collection_name, documents=rag_documents)

async def asearch(self, query, ...):
    return await client.asearch(collection_name=collection_name, query=query_text, ...)
```

从 `Knowledge.aadd_sources()` 到 `Source.aadd()` 到 `Storage.asave()`，async 调用链是完整的。这对于在 async 环境（如 FastAPI）中使用 CrewAI 至关重要。

## 15.12 CrewDoclingSource 深入分析

`CrewDoclingSource` 值得单独讨论，因为它代表了 Knowledge 系统的未来方向。

### 与普通 Source 的差异

普通文件 Source 使用简单的文本提取 + 固定窗口分块。`CrewDoclingSource` 则：

1. **文档转换**：使用 `DocumentConverter` 将各种格式转为标准的 `DoclingDocument` 对象
2. **层级分块**：使用 `HierarchicalChunker` 在文档结构边界分块

```python
def _convert_source_to_docling_documents(self) -> list[DoclingDocument]:
    conv_results_iter = self.document_converter.convert_all(self.safe_file_paths)
    return [result.document for result in conv_results_iter]

def _chunk_doc(self, doc: DoclingDocument) -> Iterator[str]:
    chunker = HierarchicalChunker()
    for chunk in chunker.chunk(doc):
        yield chunk.text
```

### 支持的格式

```python
allowed_formats=[
    InputFormat.MD,        # Markdown
    InputFormat.ASCIIDOC,  # AsciiDoc
    InputFormat.PDF,       # PDF
    InputFormat.DOCX,      # Word
    InputFormat.HTML,       # HTML
    InputFormat.IMAGE,     # 图片（OCR）
    InputFormat.XLSX,      # Excel
    InputFormat.PPTX,      # PowerPoint
]
```

八种格式，涵盖了企业环境中常见的所有文档类型。特别是图片支持（通过 OCR）和 PowerPoint 支持，是其他 Source 不具备的。

### 依赖管理

```python
try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

class CrewDoclingSource(BaseKnowledgeSource):
    def __init__(self, *args, **kwargs):
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "The docling package is required. Please install it using: uv add docling")
```

`docling` 是可选依赖，只在实际使用时才要求安装。这种模式在 CrewAI 中很常见——核心框架保持轻量，重量级依赖按需加载。

## 15.13 扩展 Knowledge Source

创建自定义知识源只需继承 `BaseKnowledgeSource` 并实现三个方法：

```python
class DatabaseKnowledgeSource(BaseKnowledgeSource):
    connection_string: str

    def validate_content(self) -> None:
        # 验证数据库连接
        pass

    def add(self) -> None:
        # 从数据库查询数据
        rows = query_database(self.connection_string)
        for row in rows:
            self.chunks.extend(self._chunk_text(str(row)))
        self._save_documents()

    async def aadd(self) -> None:
        # async 版本
        rows = await aquery_database(self.connection_string)
        for row in rows:
            self.chunks.extend(self._chunk_text(str(row)))
        await self._asave_documents()
```

核心契约是：
1. `validate_content()` 验证数据源可用性
2. `add()` / `aadd()` 负责获取数据、分块、调用 `_save_documents()`
3. `_chunk_text()` 可以重写以实现自定义分块策略

## 本章要点

- Knowledge 系统实现了完整的**文档摄入 -> 分块 -> Embedding -> 向量存储 -> 语义检索** RAG 管道
- **BaseKnowledgeSource** 定义了知识源的抽象接口，核心方法包括 `validate_content()`、`add()`、`_chunk_text()`、`_save_documents()`
- 七种内置 Source 覆盖 **PDF、CSV、Excel、JSON、Text、String、Docling 多格式**，每种负责特定格式的文件读取和内容转换
- **分块策略**默认使用固定窗口滑动（4000 字符、200 重叠），CrewDoclingSource 使用更智能的 `HierarchicalChunker`
- **KnowledgeStorage** 通过 CrewAI RAG 抽象层与 **ChromaDB** 交互，collection 命名遵循 `knowledge_{name}` 约定
- **SourceHelper** 根据文件扩展名自动路由到正确的 KnowledgeSource 类
- Knowledge 与 Memory 使用**不同的向量数据库**（ChromaDB vs LanceDB），反映了两者不同的读写模式
- 全系统支持 **async 操作**，从 `Knowledge.aadd_sources()` 到 `Storage.asave()` 形成完整的异步调用链
- `CrewDoclingSource` 支持**八种文档格式**并使用层级分块，代表了 Knowledge 系统的演进方向
- 重量级依赖（pdfplumber、pandas、docling）全部**按需动态导入**，保持框架核心轻量
