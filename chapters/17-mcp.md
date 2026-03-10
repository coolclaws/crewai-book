# 第 17 章　MCP 集成：Model Context Protocol

Model Context Protocol（MCP）是一个开放协议标准，旨在为 LLM 应用提供统一的工具和上下文接入方式。类似于 USB 为外设提供了标准接口，MCP 为 AI Agent 提供了连接外部服务的通用协议。CrewAI 对 MCP 的集成涵盖了传输层抽象、客户端管理、工具发现与包装、权限过滤等完整链路，使得任何 MCP 兼容的服务都能无缝地成为 Agent 的工具。

本章将深入剖析 `crewai/mcp/` 模块的每一个组件。

## 17.1 MCP 模块总览

```
mcp/
    __init__.py           # 统一导出所有公开 API
    config.py             # MCPServerStdio / MCPServerHTTP / MCPServerSSE 配置模型
    client.py             # MCPClient —— 核心客户端，管理 Session 生命周期
    tool_resolver.py      # MCPToolResolver —— 将 MCP 配置解析为 CrewAI 工具
    filters.py            # ToolFilter / StaticToolFilter —— 工具过滤与权限控制
    transports/
        __init__.py       # 导出所有 Transport 类
        base.py           # BaseTransport —— 传输层抽象基类
        stdio.py          # StdioTransport —— 本地进程通信
        sse.py            # SSETransport —— Server-Sent Events
        http.py           # HTTPTransport —— Streamable HTTP
```

与 tools 模块中的两个 MCP 工具包装器形成上下游关系：

```
tools/
    mcp_tool_wrapper.py   # MCPToolWrapper —— 按需连接（用于 HTTPS URL）
    mcp_native_tool.py    # MCPNativeTool —— 复用客户端（用于 Native Config）
```

整体数据流如下：

```
Agent.mcps 配置
    │
    ▼
MCPToolResolver.resolve()
    │
    ├─── MCPServerStdio  ──► StdioTransport  ──► MCPClient ──► MCPNativeTool
    ├─── MCPServerHTTP   ──► HTTPTransport   ──► MCPClient ──► MCPNativeTool
    ├─── MCPServerSSE    ──► SSETransport    ──► MCPClient ──► MCPNativeTool
    ├─── "https://..."   ──► streamablehttp  ──────────────► MCPToolWrapper
    └─── "notion"        ──► AMP API fetch   ──► MCPServerHTTP/SSE ──► ...
```

## 17.2 传输层：BaseTransport 与三种实现

传输层是 MCP 集成的底层基础，负责与 MCP Server 建立和维护通信通道。

### 17.2.1 BaseTransport 抽象基类

```python
class TransportType(str, Enum):
    STDIO = "stdio"
    HTTP = "http"
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"

class BaseTransport(ABC):
    def __init__(self, **kwargs):
        self._read_stream: ReadStream | None = None
        self._write_stream: WriteStream | None = None
        self._connected = False

    @property
    @abstractmethod
    def transport_type(self) -> TransportType: ...

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def read_stream(self) -> ReadStream:
        if self._read_stream is None:
            raise RuntimeError("Transport not connected.")
        return self._read_stream

    @abstractmethod
    async def connect(self) -> Self: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    def _set_streams(self, read, write):
        self._read_stream = read
        self._write_stream = write
        self._connected = True

    def _clear_streams(self):
        self._read_stream = None
        self._write_stream = None
        self._connected = False
```

BaseTransport 定义了清晰的传输层契约：
- **`read_stream` / `write_stream`**：MCP 协议的双向通信通道
- **`connect` / `disconnect`**：连接生命周期管理
- **`_set_streams` / `_clear_streams`**：内部辅助方法，统一管理连接状态

所有 Transport 都支持 `async with` 语法：

```python
async with StdioTransport(command="python", args=["server.py"]) as transport:
    # transport.read_stream 和 transport.write_stream 已就绪
    pass
```

### 17.2.2 StdioTransport：本地进程通信

StdioTransport 用于连接以子进程方式运行的本地 MCP Server：

```python
class StdioTransport(BaseTransport):
    def __init__(self, command, args=None, env=None, **kwargs):
        super().__init__(**kwargs)
        self.command = command
        self.args = args or []
        self.env = env or {}

    @property
    def transport_type(self) -> TransportType:
        return TransportType.STDIO

    async def connect(self) -> Self:
        if self._connected:
            return self

        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        process_env = os.environ.copy()
        process_env.update(self.env)

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=process_env if process_env else None,
        )
        self._transport_context = stdio_client(server_params)
        read, write = await self._transport_context.__aenter__()
        self._set_streams(read=read, write=write)
        return self
```

连接过程本质上是启动一个子进程，并将其 stdin/stdout 封装为 MCP 协议的读写流。环境变量通过合并当前环境和自定义变量来传递，这允许向 MCP Server 传递 API Key 等敏感配置。

断开连接时，StdioTransport 会先尝试 `terminate`，5 秒超时后强制 `kill`：

```python
async def disconnect(self):
    self._clear_streams()
    if self._transport_context is not None:
        await self._transport_context.__aexit__(None, None, None)
    if self._process is not None:
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self._process.kill()
```

### 17.2.3 HTTPTransport：Streamable HTTP

HTTPTransport 使用 MCP SDK 提供的 `streamablehttp_client` 进行远程通信：

```python
class HTTPTransport(BaseTransport):
    def __init__(self, url, headers=None, streamable=True, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self.headers = headers or {}
        self.streamable = streamable

    @property
    def transport_type(self) -> TransportType:
        return TransportType.STREAMABLE_HTTP if self.streamable else TransportType.HTTP

    async def connect(self) -> Self:
        from mcp.client.streamable_http import streamablehttp_client

        self._transport_context = streamablehttp_client(
            self.url,
            headers=self.headers if self.headers else None,
            terminate_on_close=True,
        )
        read, write, _ = await asyncio.wait_for(
            self._transport_context.__aenter__(), timeout=30.0
        )
        self._set_streams(read=read, write=write)
        return self
```

注意 `terminate_on_close=True` 参数确保连接关闭时自动清理后台 task。`disconnect` 方法中包含了对 anyio `cancel scope` 错误的精细处理 —— 在 asyncio 和 anyio 混合使用场景下，退出 transport 上下文时可能触发 `BaseExceptionGroup`，代码会判断其中是否只包含 cancel scope / task 相关的错误，如果是则安全忽略，否则重新抛出。

### 17.2.4 SSETransport：Server-Sent Events

SSETransport 使用 MCP SDK 的 `sse_client` 实现实时流式通信：

```python
class SSETransport(BaseTransport):
    def __init__(self, url, headers=None, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self.headers = headers or {}

    @property
    def transport_type(self) -> TransportType:
        return TransportType.SSE

    async def connect(self) -> Self:
        from mcp.client.sse import sse_client

        self._transport_context = sse_client(
            self.url,
            headers=self.headers if self.headers else None,
        )
        read, write = await self._transport_context.__aenter__()
        self._set_streams(read=read, write=write)
        return self
```

SSE 传输相对最简单，因为 SSE 协议本身就是单向推送模型，通信复杂性由 MCP SDK 内部处理。

### 17.2.5 三种传输的对比

| 特性 | StdioTransport | HTTPTransport | SSETransport |
|------|---------------|---------------|-------------|
| 连接方式 | 本地子进程 | HTTP/HTTPS | Server-Sent Events |
| 适用场景 | 本地开发、CLI 工具 | 远程 API 服务 | 实时流式服务 |
| 延迟 | 极低（进程间） | 中等（网络） | 中等（网络） |
| 认证方式 | 环境变量 | HTTP Headers | HTTP Headers |
| 连接超时 | N/A | 30 秒 | 依赖 SDK |
| 特殊处理 | 进程生命周期 | cancel scope 错误 | 无 |

## 17.3 MCPClient：核心客户端

MCPClient 是 MCP 集成的中枢，管理与 MCP Server 的 Session 生命周期，提供工具列表、工具调用、Prompt 管理等高级 API。

### 17.3.1 初始化与连接管理

```python
class MCPClient:
    def __init__(self, transport, connect_timeout=30,
                 execution_timeout=30, discovery_timeout=30,
                 max_retries=3, cache_tools_list=False, logger=None):
        self.transport = transport
        self.connect_timeout = connect_timeout
        self.execution_timeout = execution_timeout
        self.discovery_timeout = discovery_timeout
        self.max_retries = max_retries
        self.cache_tools_list = cache_tools_list
        self._session = None
        self._initialized = False
        self._exit_stack = AsyncExitStack()
        self._was_connected = False
```

`AsyncExitStack` 是一个关键的设计选择。它将 Transport 和 ClientSession 的异步上下文管理器统一管理在同一个栈中，确保它们在同一个异步作用域内创建和销毁，避免了 anyio 中 "exit cancel scope in different task" 的错误。

`connect` 方法的核心流程：

```python
async def connect(self) -> Self:
    if self.connected:
        return self

    # 发出 MCPConnectionStartedEvent
    try:
        from mcp import ClientSession

        # 通过 exit stack 管理 transport 上下文
        await self._exit_stack.enter_async_context(self.transport)

        # 创建并管理 ClientSession
        self._session = ClientSession(
            self.transport.read_stream,
            self.transport.write_stream,
        )
        await self._exit_stack.enter_async_context(self._session)

        # 初始化 MCP 协议握手
        await asyncio.wait_for(
            self._session.initialize(),
            timeout=self.connect_timeout,
        )

        self._initialized = True
        self._was_connected = True
        # 发出 MCPConnectionCompletedEvent
        return self

    except asyncio.TimeoutError:
        await self._cleanup_on_error()
        raise ConnectionError(f"MCP connection timed out...")
    except BaseExceptionGroup as eg:
        # 从异常组中提取真正的错误
        actual_error = None
        for exc in eg.exceptions:
            if isinstance(exc, Exception) and not isinstance(exc, GeneratorExit):
                error_msg = str(exc).lower()
                if "401" in error_msg or "unauthorized" in error_msg:
                    actual_error = exc
                    break
        # ...
```

异常处理中对 `BaseExceptionGroup` 的特殊处理值得关注。在 Python 3.11+ 中，anyio 的 task group 可能将多个异常打包为异常组。代码从中提取有意义的错误（如 401 认证失败），过滤掉 `GeneratorExit` 和 cancel scope 相关的噪音，确保用户看到的是真正的根因而非框架内部的清理错误。

### 17.3.2 工具列表与缓存

```python
# 模块级缓存
_mcp_schema_cache: dict[str, tuple[dict[str, Any], float]] = {}
_cache_ttl = 300  # 5 分钟

async def list_tools(self, use_cache=None):
    if not self.connected:
        await self.connect()

    use_cache = use_cache if use_cache is not None else self.cache_tools_list
    if use_cache:
        cache_key = self._get_cache_key("tools")
        if cache_key in _mcp_schema_cache:
            cached_data, cache_time = _mcp_schema_cache[cache_key]
            if time.time() - cache_time < _cache_ttl:
                return cached_data

    tools = await self._retry_operation(
        self._list_tools_impl,
        timeout=self.discovery_timeout,
    )

    if use_cache:
        _mcp_schema_cache[cache_key] = (tools, time.time())

    return tools
```

缓存策略使用模块级的 `_mcp_schema_cache` 字典，TTL 为 5 分钟。缓存键通过 `_get_cache_key` 生成，格式因传输类型而异：

```python
def _get_cache_key(self, resource_type):
    if isinstance(self.transport, StdioTransport):
        key = f"stdio:{self.transport.command}:{':'.join(self.transport.args)}"
    elif isinstance(self.transport, HTTPTransport):
        key = f"http:{self.transport.url}"
    elif isinstance(self.transport, SSETransport):
        key = f"sse:{self.transport.url}"
    return f"mcp:{key}:{resource_type}"
```

`_list_tools_impl` 从 MCP Session 获取原始工具列表，并进行名称标准化：

```python
async def _list_tools_impl(self):
    tools_result = await asyncio.wait_for(
        self.session.list_tools(),
        timeout=self.discovery_timeout,
    )
    return [
        {
            "name": sanitize_tool_name(tool.name),
            "original_name": tool.name,
            "description": getattr(tool, "description", ""),
            "inputSchema": getattr(tool, "inputSchema", {}),
        }
        for tool in tools_result.tools
    ]
```

注意 `name` 经过了 `sanitize_tool_name` 处理（去除特殊字符、统一格式），但同时保留了 `original_name` 用于实际调用 MCP Server 时的匹配。

### 17.3.3 工具调用

`call_tool` 方法封装了完整的工具调用流程：

```python
async def call_tool(self, tool_name, arguments=None):
    if not self.connected:
        await self.connect()

    arguments = arguments or {}
    cleaned_arguments = self._clean_tool_arguments(arguments)

    # 发出 MCPToolExecutionStartedEvent
    try:
        tool_result: _MCPToolResult = await self._retry_operation(
            lambda: self._call_tool_impl(tool_name, cleaned_arguments),
            timeout=self.execution_timeout,
        )

        if tool_result.is_error:
            # 发出 MCPToolExecutionFailedEvent
        else:
            # 发出 MCPToolExecutionCompletedEvent

        return tool_result.content
    except Exception as e:
        # 发出 MCPToolExecutionFailedEvent
        raise
```

`_MCPToolResult` 是一个 NamedTuple，携带了 MCP 协议中的 `isError` 标志：

```python
class _MCPToolResult(NamedTuple):
    content: str
    is_error: bool
```

`_clean_tool_arguments` 方法对参数进行清理，包括移除 `None` 值、修复 `sources` 数组格式、递归清理嵌套字典：

```python
def _clean_tool_arguments(self, arguments):
    cleaned = {}
    for key, value in arguments.items():
        if value is None:
            continue
        # 修复 sources: ["web"] -> [{"type": "web"}]
        if key == "sources" and isinstance(value, list):
            fixed_sources = []
            for item in value:
                if isinstance(item, str):
                    fixed_sources.append({"type": item})
                elif isinstance(item, dict):
                    fixed_sources.append(item)
            if fixed_sources:
                cleaned[key] = fixed_sources
            continue
        # 递归处理嵌套结构
        if isinstance(value, dict):
            nested = self._clean_tool_arguments(value)
            if nested:
                cleaned[key] = nested
        elif isinstance(value, list):
            # ... 清理列表元素
        else:
            cleaned[key] = value
    return cleaned
```

### 17.3.4 重试机制

`_retry_operation` 实现了通用的指数退避重试：

`_retry_operation` 的核心设计是**错误分类**：认证失败（`authentication` / `unauthorized`）和资源未找到（`not found`）是确定性错误，直接抛出不重试；超时和其他错误可能是暂时性的，使用 2^attempt 秒的指数退避重试，最多 `max_retries`（默认 3）次。

## 17.4 MCP 配置模型

`config.py` 定义了三种 MCP Server 配置，全部基于 Pydantic BaseModel：

```python
class MCPServerStdio(BaseModel):
    command: str       # 执行命令，如 "python", "node", "npx"
    args: list[str] = []
    env: dict[str, str] | None = None
    tool_filter: ToolFilter | None = None
    cache_tools_list: bool = False

class MCPServerHTTP(BaseModel):
    url: str           # 如 "https://api.example.com/mcp"
    headers: dict[str, str] | None = None
    streamable: bool = True
    tool_filter: ToolFilter | None = None
    cache_tools_list: bool = False

class MCPServerSSE(BaseModel):
    url: str           # 如 "https://api.example.com/mcp/sse"
    headers: dict[str, str] | None = None
    tool_filter: ToolFilter | None = None
    cache_tools_list: bool = False

MCPServerConfig = MCPServerStdio | MCPServerHTTP | MCPServerSSE
```

每种配置都内置了 `tool_filter` 和 `cache_tools_list` 字段，这使得过滤和缓存可以在最接近配置定义的位置声明。

使用示例：

```python
from crewai.mcp import MCPServerStdio, MCPServerHTTP, MCPServerSSE

# 本地文件系统服务器
fs_server = MCPServerStdio(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    env={"NODE_OPTIONS": "--max-old-space-size=4096"},
)

# 远程 HTTP 服务器
api_server = MCPServerHTTP(
    url="https://api.example.com/mcp",
    headers={"Authorization": "Bearer sk-xxx"},
    cache_tools_list=True,
)

# SSE 服务器
sse_server = MCPServerSSE(
    url="https://stream.example.com/mcp/sse",
    headers={"X-API-Key": "xxx"},
)
```

## 17.5 ToolFilter：工具过滤与权限控制

`filters.py` 提供了两种工具过滤机制：静态过滤和动态过滤。

### 17.5.1 StaticToolFilter

```python
class StaticToolFilter:
    def __init__(self, allowed_tool_names=None, blocked_tool_names=None):
        self.allowed_tool_names = set(allowed_tool_names or [])
        self.blocked_tool_names = set(blocked_tool_names or [])

    def __call__(self, tool: dict[str, Any]) -> bool:
        tool_name = tool.get("name", "")
        # 黑名单优先
        if self.blocked_tool_names and tool_name in self.blocked_tool_names:
            return False
        # 白名单校验
        if self.allowed_tool_names:
            return tool_name in self.allowed_tool_names
        # 无限制
        return True
```

黑名单（`blocked_tool_names`）优先于白名单（`allowed_tool_names`），这符合安全最佳实践中 "deny by default" 的原则。便捷的工厂函数：

```python
filter_fn = create_static_tool_filter(
    allowed_tool_names=["read_file", "write_file"],
    blocked_tool_names=["delete_file"],
)
```

### 17.5.2 动态过滤与 ToolFilterContext

对于需要根据运行时上下文决定工具可用性的场景，框架提供了 `ToolFilterContext`：

```python
class ToolFilterContext(BaseModel):
    agent: Any
    server_name: str
    run_context: dict[str, Any] | None = None
```

动态过滤器可以访问当前 Agent 的信息和运行上下文：

```python
def context_aware_filter(context: ToolFilterContext, tool: dict) -> bool:
    # Code Reviewer 不能使用危险工具
    if context.agent.role == "Code Reviewer":
        if tool["name"].startswith("danger_"):
            return False
    return True

filter_fn = create_dynamic_tool_filter(context_aware_filter)

mcp_server = MCPServerStdio(
    command="python",
    args=["server.py"],
    tool_filter=filter_fn,
)
```

`ToolFilter` 类型别名支持两种签名：

```python
ToolFilter = (
    Callable[[ToolFilterContext, dict[str, Any]], bool]   # 带上下文
    | Callable[[dict[str, Any]], bool]                     # 无上下文
)
```

MCPToolResolver 在应用过滤器时会自动检测签名并适配调用方式：

```python
if callable(mcp_config.tool_filter):
    try:
        context = ToolFilterContext(agent=self._agent, server_name=server_name)
        if mcp_config.tool_filter(context, tool):
            filtered_tools.append(tool)
    except (TypeError, AttributeError):
        # 降级为无上下文的调用
        if mcp_config.tool_filter(tool):
            filtered_tools.append(tool)
```

## 17.6 MCPToolResolver：工具解析引擎

`MCPToolResolver` 是连接 Agent 配置与 MCP 工具的桥梁，负责将各种形式的 MCP 引用解析为 CrewAI 工具实例。

### 17.6.1 三种 MCP 引用形式

MCPToolResolver 支持三种不同的 MCP 引用方式：

1. **Native Config**：`MCPServerStdio` / `MCPServerHTTP` / `MCPServerSSE` 对象
2. **HTTPS URL**：如 `"https://mcp.example.com/api"`
3. **AMP 引用**：如 `"notion"` 或 `"notion#search"`（CrewAI 平台的托管 MCP 服务）

`resolve` 方法根据输入类型分发处理：

```python
class MCPToolResolver:
    def __init__(self, agent, logger):
        self._agent = agent
        self._logger = logger
        self._clients: list[Any] = []

    def resolve(self, mcps: list[str | MCPServerConfig]) -> list[BaseTool]:
        all_tools = []
        amp_refs = []

        for mcp_config in mcps:
            if isinstance(mcp_config, str) and mcp_config.startswith("https://"):
                all_tools.extend(self._resolve_external(mcp_config))
            elif isinstance(mcp_config, str):
                amp_refs.append(self._parse_amp_ref(mcp_config))
            else:
                tools, client = self._resolve_native(mcp_config)
                all_tools.extend(tools)
                if client:
                    self._clients.append(client)

        if amp_refs:
            tools, clients = self._resolve_amp(amp_refs)
            all_tools.extend(tools)
            self._clients.extend(clients)

        return all_tools
```

### 17.6.2 Native Config 解析

`_resolve_native` 是最完整的解析路径，包含五个步骤：

1. **创建 Transport**：根据配置类型（Stdio / HTTP / SSE）实例化对应的 Transport
2. **创建 MCPClient**：传入 transport 和缓存配置
3. **连接并发现工具**：通过 `asyncio.run()` 执行异步的连接、工具列表获取和断开操作
4. **应用工具过滤器**：如果配置了 `tool_filter`，自动检测签名类型并过滤
5. **转换为 MCPNativeTool**：将每个工具定义封装为 CrewAI 的 BaseTool 子类

其中步骤 5 的关键操作是 `_json_schema_to_pydantic`，它将 MCP Server 返回的 JSON Schema 转换为 Pydantic 模型：

```python
@staticmethod
def _json_schema_to_pydantic(tool_name, json_schema):
    from crewai.utilities.pydantic_schema_utils import create_model_from_schema
    model_name = f"{tool_name.replace('-', '_').replace(' ', '_')}Schema"
    return create_model_from_schema(json_schema, model_name=model_name, enrich_descriptions=True)
```

这是让 MCP 工具融入 CrewAI 类型系统的关键桥梁。

### 17.6.3 HTTPS URL 解析

对于纯 URL 形式的 MCP 引用，使用 MCPToolWrapper（按需连接方式）：

```python
def _resolve_external(self, mcp_ref):
    if "#" in mcp_ref:
        server_url, specific_tool = mcp_ref.split("#", 1)
    else:
        server_url, specific_tool = mcp_ref, None

    server_params = {"url": server_url}
    server_name = self._extract_server_name(server_url)
    tool_schemas = self._get_mcp_tool_schemas(server_params)

    tools = []
    for tool_name, schema in tool_schemas.items():
        if specific_tool and tool_name != specific_tool:
            continue
        wrapper = MCPToolWrapper(
            mcp_server_params=server_params,
            tool_name=tool_name,
            tool_schema=schema,
            server_name=server_name,
        )
        tools.append(wrapper)
    return tools
```

URL 中的 `#` 语法允许选择特定工具：`"https://mcp.example.com/api#search"` 只获取名为 `search` 的工具。

### 17.6.4 AMP 引用解析

AMP（Agent Marketplace Platform）引用是 CrewAI 平台特有的功能，通过 slug 引用托管的 MCP 服务。`_resolve_amp` 的处理流程：

1. 去重 slug 列表（同一 Server 只连接一次）
2. 通过 CrewAI+ API 批量获取 MCP 配置（`_fetch_amp_mcp_configs`）
3. 将返回的配置字典转换为 `MCPServerHTTP` 或 `MCPServerSSE`（`_build_mcp_config_from_dict`）
4. 复用 `_resolve_native` 解析为工具
5. 按照原始引用中的 `#specific_tool` 后缀过滤特定工具

AMP slug 支持 `"notion"` 形式获取所有工具，也支持 `"notion#search"` 形式只获取特定工具。还兼容旧版的 `"crewai-amp:notion"` 前缀格式。

### 17.6.5 清理与资源管理

```python
def cleanup(self):
    if not self._clients:
        return

    async def _disconnect_all():
        for client in self._clients:
            if client and hasattr(client, "connected") and client.connected:
                await client.disconnect()

    try:
        asyncio.run(_disconnect_all())
    except Exception as e:
        self._logger.log("error", f"Error during MCP client cleanup: {e}")
    finally:
        self._clients.clear()
```

MCPToolResolver 持有所有创建的 MCPClient 引用，在 `cleanup` 时统一断开连接。这种集中式资源管理避免了连接泄漏。

## 17.7 MCP 工具包装器

### 17.7.1 MCPToolWrapper：按需连接

MCPToolWrapper 用于 HTTPS URL 形式的 MCP 引用，每次调用时建立新连接：

```python
class MCPToolWrapper(BaseTool):
    def __init__(self, mcp_server_params, tool_name, tool_schema, server_name):
        prefixed_name = f"{server_name}_{tool_name}"
        super().__init__(name=prefixed_name, description=..., args_schema=...)
        self._mcp_server_params = mcp_server_params
        self._original_tool_name = tool_name

    def _run(self, **kwargs) -> str:
        try:
            return asyncio.run(self._run_async(**kwargs))
        except asyncio.TimeoutError:
            return f"MCP tool '{self.original_tool_name}' timed out..."
        except Exception as e:
            return f"Error executing MCP tool {self.original_tool_name}: {e!s}"
```

核心执行逻辑包含指数退避重试（`_retry_with_exponential_backoff`），最多 3 次尝试，等待时间为 2^attempt 秒。错误分类与 MCPClient 一致：认证错误和工具未找到不重试，超时和网络错误重试。

### 17.7.2 MCPNativeTool：复用 Session

MCPNativeTool 用于 Native Config 形式的 MCP 引用，持有 MCPClient 实例的引用。其 `_run_async` 方法的核心逻辑：

```python
async def _run_async(self, **kwargs):
    # asyncio.run() 每次创建新 event loop，transport 上下文无法跨 loop
    if self._mcp_client.connected:
        await self._mcp_client.disconnect()
    await self._mcp_client.connect()
    try:
        result = await self._mcp_client.call_tool(self.original_tool_name, kwargs)
    except Exception as e:
        if "not connected" in str(e).lower() or "connection" in str(e).lower():
            await self._mcp_client.disconnect()
            await self._mcp_client.connect()
            result = await self._mcp_client.call_tool(self.original_tool_name, kwargs)
        else:
            raise
    finally:
        await self._mcp_client.disconnect()  # 确保 transport 上下文生命周期完整
    return result
```

代码注释中解释了一个关键的技术限制：`asyncio.run()` 每次创建新的 event loop，而 MCP transport 的 context manager（基于 anyio task group）无法跨 event loop 使用，所以即使是 "Native" 工具，每次调用也需要重新连接。`_run` 方法还处理了嵌套 event loop 的情况 —— 如果已有运行中的 loop，则在 `ThreadPoolExecutor` 中另起一个 `asyncio.run()`。

## 17.8 事件驱动的可观测性

MCP 模块通过 `crewai_event_bus` 发出丰富的事件，贯穿整个连接和执行生命周期：

| 事件 | 触发时机 |
|------|---------|
| `MCPConnectionStartedEvent` | 开始连接 MCP Server |
| `MCPConnectionCompletedEvent` | 连接成功，包含耗时信息 |
| `MCPConnectionFailedEvent` | 连接失败，包含错误类型和详情 |
| `MCPToolExecutionStartedEvent` | 开始执行 MCP 工具 |
| `MCPToolExecutionCompletedEvent` | 工具执行成功，包含执行耗时 |
| `MCPToolExecutionFailedEvent` | 工具执行失败，包含错误分类 |
| `MCPConfigFetchFailedEvent` | AMP 配置获取失败 |

这些事件携带了足够的上下文信息（server_name、transport_type、tool_name、duration_ms 等），使得外部监控系统可以构建完整的 MCP 调用链追踪。

## 17.9 完整集成示例

将所有组件组合在一起，一个典型的 MCP 集成配置如下：

```python
from crewai import Agent, Task, Crew
from crewai.mcp import (
    MCPServerStdio,
    MCPServerHTTP,
    create_static_tool_filter,
    create_dynamic_tool_filter,
    ToolFilterContext,
)

# 本地文件系统 MCP Server
fs_server = MCPServerStdio(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
    tool_filter=create_static_tool_filter(
        allowed_tool_names=["read_file", "write_file", "list_directory"],
        blocked_tool_names=["delete_file"],
    ),
)

# 远程搜索 MCP Server
search_server = MCPServerHTTP(
    url="https://search.example.com/mcp",
    headers={"Authorization": "Bearer sk-xxx"},
    cache_tools_list=True,
)

# 动态过滤：研究员可以搜索，但不能修改
def researcher_filter(ctx: ToolFilterContext, tool: dict) -> bool:
    if ctx.agent.role == "Researcher":
        return not tool["name"].startswith("write_")
    return True

restricted_server = MCPServerHTTP(
    url="https://api.example.com/mcp",
    headers={"Authorization": "Bearer sk-xxx"},
    tool_filter=create_dynamic_tool_filter(researcher_filter),
)

# 创建 Agent
researcher = Agent(
    role="Researcher",
    goal="Find relevant information",
    backstory="An expert researcher...",
    mcps=[fs_server, search_server, restricted_server],
)

# 也支持 URL 直接引用（按需连接）
analyst = Agent(
    role="Analyst",
    goal="Analyze data",
    backstory="A data analyst...",
    mcps=["https://data.example.com/mcp"],
)

# AMP 平台引用
writer = Agent(
    role="Writer",
    goal="Write reports",
    backstory="A technical writer...",
    mcps=["notion", "google-docs#create_document"],
)
```

## 17.10 设计洞察与架构权衡

### 连接即断即连的取舍

MCPNativeTool 虽然名为 "Native"（原生），却不得不在每次工具调用时重新建立连接。这看似低效，实则是对 Python asyncio 与 anyio 交互限制的务实妥协。MCP SDK 的 transport context manager 基于 anyio 的 task group，而 task group 必须在同一个 event loop 中创建和关闭。由于 CrewAI 的工具调用可能发生在不同的 `asyncio.run()` 调用中（各自创建独立的 event loop），持久连接在技术上不可行。

### 三层工具发现的统一

MCPToolResolver 将三种截然不同的 MCP 引用方式（Native Config、HTTPS URL、AMP slug）统一到了同一个 `resolve` 接口下。这种统一使得 Agent 配置可以自由混合不同类型的 MCP 引用，而无需关心底层的连接和发现机制。

### 安全性的分层设计

MCP 集成的安全性通过多层机制保障：
- **传输层**：HTTP headers 支持 Bearer token、API key 等认证
- **过滤层**：StaticToolFilter 和动态过滤器控制工具可见性
- **执行层**：ToolUsage 注入 fingerprint 安全上下文
- **协议层**：MCP 协议自身的错误标志（`isError`）传播

## 本章要点

- **MCP 传输层**提供三种实现：StdioTransport（本地进程通信）、HTTPTransport（Streamable HTTP 远程通信）、SSETransport（Server-Sent Events），统一继承自 BaseTransport 抽象基类
- **MCPClient** 是核心客户端类，使用 `AsyncExitStack` 管理 Transport 和 ClientSession 的生命周期，内置 5 分钟 TTL 的工具列表缓存和指数退避重试机制
- **MCP 配置模型** `MCPServerStdio` / `MCPServerHTTP` / `MCPServerSSE` 提供类型安全的服务器配置，每种配置都内置 `tool_filter` 和 `cache_tools_list` 支持
- **ToolFilter** 支持静态过滤（allow/block list，黑名单优先）和动态过滤（基于 ToolFilterContext 的运行时决策），通过 `create_static_tool_filter` / `create_dynamic_tool_filter` 工厂函数创建
- **MCPToolResolver** 统一解析三种 MCP 引用形式：Native Config 通过 MCPClient + MCPNativeTool，HTTPS URL 通过 MCPToolWrapper 按需连接，AMP slug 通过 CrewAI+ API 获取配置后转为 Native Config 处理
- **MCPToolWrapper** 每次调用建立新连接，适用于 HTTPS URL 引用；**MCPNativeTool** 持有 MCPClient 引用但因 event loop 限制每次仍需重连
- `_clean_tool_arguments` 方法递归清理工具参数，移除 `None` 值并修复特殊格式（如 sources 数组）
- MCP 模块通过事件总线发出 7 种事件，覆盖连接、执行、配置获取的完整生命周期，支持外部监控和追踪
- JSON Schema 到 Pydantic 模型的自动转换（`_json_schema_to_pydantic`）是 MCP 工具融入 CrewAI 类型系统的关键桥梁
