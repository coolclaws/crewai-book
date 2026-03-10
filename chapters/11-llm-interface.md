# 第 11 章　统一 LLM 接口

大型语言模型的提供商众多，API 格式各异。OpenAI 用 `messages` + `tools`，Anthropic 要求首条消息必须为 `user` 角色，Bedrock 走 AWS IAM 认证，Gemini 有独立的 `ThinkingConfig`。如果每个 Agent 都要自己处理这些差异，框架的可维护性将急剧下降。

CrewAI 通过 `BaseLLM` 抽象基类和 `LLM` 工厂类构建了一套统一的 LLM 接口层，使上层代码（Agent、Task、Crew）完全不需要感知底层 Provider 的差异。本章将深入分析这一架构的设计与实现。

## 11.1 架构总览

LLM 接口层的文件结构如下：

```
crewai/
├── llm.py                          # LLM 工厂类（~2400 行）
└── llms/
    ├── base_llm.py                 # BaseLLM 抽象基类
    ├── constants.py                # 各 Provider 的模型常量列表
    ├── hooks/                      # 传输层拦截器
    │   ├── base.py                 # BaseInterceptor 抽象类
    │   └── transport.py            # HTTP 传输层实现
    ├── providers/                  # 原生 SDK 实现
    │   ├── openai/completion.py    # OpenAICompletion
    │   ├── anthropic/completion.py # AnthropicCompletion
    │   ├── gemini/completion.py    # GeminiCompletion
    │   ├── bedrock/completion.py   # BedrockCompletion
    │   └── azure/completion.py     # AzureCompletion
    └── third_party/                # 第三方模型集成
        └── __init__.py
```

核心设计思想是**两级路由**：`LLM` 类的 `__new__` 方法作为工厂，根据模型名称自动路由到原生 SDK Provider；如果没有匹配的原生 Provider，则回退到 LiteLLM 作为统一的后备方案。

## 11.2 BaseLLM 抽象基类

`BaseLLM` 定义在 `llms/base_llm.py` 中，是所有 LLM 实现的公共接口。它是一个纯 Python 类（非 Pydantic Model），通过抽象方法和模板方法模式为各个 Provider 提供统一的行为契约。

### 11.2.1 类定义与初始化

```python
class BaseLLM(ABC):
    is_litellm: bool = False

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        prefer_upload: bool = False,
        **kwargs: Any,
    ) -> None:
        if not model:
            raise ValueError("Model name is required and cannot be empty")

        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        self.prefer_upload = prefer_upload
        self.additional_params = kwargs
        self._provider = provider or "openai"

        # 统一的 stop words 处理
        stop = kwargs.pop("stop", None)
        if stop is None:
            self.stop: list[str] = []
        elif isinstance(stop, str):
            self.stop = [stop]
        elif isinstance(stop, list):
            self.stop = stop
        else:
            self.stop = []

        # Token 使用量追踪
        self._token_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "successful_requests": 0,
            "cached_prompt_tokens": 0,
        }
```

初始化方法做了几件关键的事：

1. **强制要求 model 参数**：空字符串会立即抛出 `ValueError`
2. **统一 stop words 格式**：无论传入 `str`、`list` 还是 `None`，都归一化为 `list[str]`
3. **内建 token 使用量追踪**：每个 LLM 实例独立维护自己的 token 消耗统计
4. **Provider 默认值**：未指定时默认为 `"openai"`

### 11.2.2 核心抽象方法

`BaseLLM` 定义了一个核心抽象方法 `call`，以及一个带默认实现的异步版本 `acall`：

```python
@abstractmethod
def call(
    self,
    messages: str | list[LLMMessage],
    tools: list[dict[str, BaseTool]] | None = None,
    callbacks: list[Any] | None = None,
    available_functions: dict[str, Any] | None = None,
    from_task: Task | None = None,
    from_agent: Agent | None = None,
    response_model: type[BaseModel] | None = None,
) -> str | Any:
    """调用 LLM 并返回响应"""

async def acall(
    self,
    messages: str | list[LLMMessage],
    ...
) -> str | Any:
    """异步调用 LLM，默认抛出 NotImplementedError"""
    raise NotImplementedError
```

`call` 方法的参数设计非常精心：

- **messages**：支持字符串（自动包装为 user message）和标准消息列表两种格式
- **tools**：标准化的工具 schema 列表，各 Provider 在内部转换为自己的格式
- **available_functions**：函数名到可调用对象的映射，用于实际执行工具调用
- **from_task / from_agent**：调用者上下文，用于事件追踪和 Hook 系统
- **response_model**：Pydantic 模型类，用于结构化输出

### 11.2.3 模板方法：事件发射

`BaseLLM` 提供了一组事件发射的模板方法，确保所有 Provider 以一致的方式报告生命周期事件：

```python
def _emit_call_started_event(self, messages, tools, ...) -> None:
    crewai_event_bus.emit(
        self,
        event=LLMCallStartedEvent(
            messages=to_serializable(messages),
            model=self.model,
            call_id=get_current_call_id(),
            ...
        ),
    )

def _emit_call_completed_event(self, response, call_type, ...) -> None:
    crewai_event_bus.emit(
        self,
        event=LLMCallCompletedEvent(
            response=to_serializable(response),
            call_type=call_type,
            model=self.model,
            call_id=get_current_call_id(),
            ...
        ),
    )

def _emit_call_failed_event(self, error, ...) -> None:
    ...

def _emit_stream_chunk_event(self, chunk, ...) -> None:
    ...

def _emit_thinking_chunk_event(self, chunk, ...) -> None:
    ...
```

`call_id` 通过 `contextvars` 实现跨调用链的追踪：

```python
_current_call_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_call_id", default=None
)

@contextmanager
def llm_call_context() -> Generator[str, None, None]:
    call_id = str(uuid.uuid4())
    token = _current_call_id.set(call_id)
    try:
        yield call_id
    finally:
        _current_call_id.reset(token)
```

这种设计确保了即使在并发场景下，每次 LLM 调用都能拥有唯一的 `call_id`，从而实现精确的事件关联。

### 11.2.4 Token 使用量追踪

`BaseLLM` 内建了跨 Provider 的 token 统计方法：

```python
def _track_token_usage_internal(self, usage_data: dict[str, Any]) -> None:
    prompt_tokens = (
        usage_data.get("prompt_tokens")
        or usage_data.get("prompt_token_count")    # Gemini 格式
        or usage_data.get("input_tokens")           # Anthropic 格式
        or 0
    )
    completion_tokens = (
        usage_data.get("completion_tokens")
        or usage_data.get("candidates_token_count")  # Gemini 格式
        or usage_data.get("output_tokens")            # Anthropic 格式
        or 0
    )
    ...
```

各个 Provider 对 token 使用量字段的命名不同，这个方法通过 fallback 链统一抽取。最终通过 `get_token_usage_summary()` 返回 `UsageMetrics` Pydantic 模型：

```python
class UsageMetrics(BaseModel):
    total_tokens: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0

    def add_usage_metrics(self, usage_metrics: Self) -> None:
        self.total_tokens += usage_metrics.total_tokens
        ...
```

### 11.2.5 Stop Words 统一处理

不同 Provider 对 stop words 的支持各异。`BaseLLM` 提供了一个 `_apply_stop_words` 方法，在 Provider 不原生支持 stop words 时进行后处理截断：

```python
def _apply_stop_words(self, content: str) -> str:
    if not self.stop or not content:
        return content

    earliest_stop_pos = len(content)
    found_stop_word = None

    for stop_word in self.stop:
        stop_pos = content.find(stop_word)
        if stop_pos != -1 and stop_pos < earliest_stop_pos:
            earliest_stop_pos = stop_pos
            found_stop_word = stop_word

    if found_stop_word is not None:
        return content[:earliest_stop_pos].strip()
    return content
```

## 11.3 LLM 工厂类与路由机制

`LLM` 类继承自 `BaseLLM`，其核心创新在于 `__new__` 方法实现的工厂模式——同一个类的构造可能返回完全不同类型的实例。

### 11.3.1 路由决策逻辑

```python
class LLM(BaseLLM):
    def __new__(cls, model: str, is_litellm: bool = False, **kwargs: Any) -> LLM:
        explicit_provider = kwargs.get("provider")

        if explicit_provider:
            provider = explicit_provider
            use_native = True
            model_string = model
        elif "/" in model:
            prefix, _, model_part = model.partition("/")
            provider_mapping = {
                "openai": "openai", "anthropic": "anthropic",
                "claude": "anthropic", "azure": "azure",
                "azure_openai": "azure", "google": "gemini",
                "gemini": "gemini", "bedrock": "bedrock",
                "aws": "bedrock",
            }
            canonical_provider = provider_mapping.get(prefix.lower())
            if canonical_provider and cls._validate_model_in_constants(
                model_part, canonical_provider
            ):
                provider = canonical_provider
                use_native = True
                model_string = model_part
            else:
                provider = prefix
                use_native = False
                model_string = model_part
        else:
            provider = cls._infer_provider_from_model(model)
            use_native = True
            model_string = model
```

路由优先级为：

1. **显式 provider 参数**：`LLM("gpt-4o", provider="openai")` 直接使用指定 Provider
2. **斜杠分隔的模型名**：`LLM("anthropic/claude-3-5-sonnet-20241022")` 从前缀推断 Provider
3. **纯模型名推断**：`LLM("claude-3-5-sonnet-20241022")` 在常量表中查找匹配的 Provider

### 11.3.2 原生 Provider 映射

```python
SUPPORTED_NATIVE_PROVIDERS: Final[list[str]] = [
    "openai", "anthropic", "claude",
    "azure", "azure_openai",
    "google", "gemini",
    "bedrock", "aws",
]

@classmethod
def _get_native_provider(cls, provider: str) -> type | None:
    if provider == "openai":
        from crewai.llms.providers.openai.completion import OpenAICompletion
        return OpenAICompletion
    if provider == "anthropic" or provider == "claude":
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion
        return AnthropicCompletion
    if provider == "azure" or provider == "azure_openai":
        from crewai.llms.providers.azure.completion import AzureCompletion
        return AzureCompletion
    if provider == "google" or provider == "gemini":
        from crewai.llms.providers.gemini.completion import GeminiCompletion
        return GeminiCompletion
    if provider == "bedrock":
        from crewai.llms.providers.bedrock.completion import BedrockCompletion
        return BedrockCompletion
    return None
```

注意所有 import 都是延迟加载的——只有实际使用某个 Provider 时才会 import 对应的 SDK。这避免了安装全部依赖的要求。

### 11.3.3 LiteLLM 回退

如果模型没有匹配到任何原生 Provider，或者原生 Provider 初始化失败，`LLM` 会回退到 LiteLLM：

```python
if not LITELLM_AVAILABLE:
    native_list = ", ".join(SUPPORTED_NATIVE_PROVIDERS)
    error_msg = (
        f"Unable to initialize LLM with model '{model}'. "
        f"The model did not match any supported native provider "
        f"({native_list}), and the LiteLLM fallback package is not installed.\n\n"
        f"To fix this, either:\n"
        f"  1. Install LiteLLM: uv add 'crewai[litellm]'\n"
        ...
    )
    raise ImportError(error_msg) from None

instance = object.__new__(cls)
super(LLM, instance).__init__(model=model, is_litellm=True, **kwargs)
instance.is_litellm = True
return instance
```

## 11.4 五大原生 Provider 实现

### 11.4.1 OpenAICompletion

`OpenAICompletion` 是功能最丰富的原生 Provider，支持两套 API：

```python
class OpenAICompletion(BaseLLM):
    BUILTIN_TOOL_TYPES: ClassVar[dict[str, str]] = {
        "web_search": "web_search_preview",
        "file_search": "file_search",
        "code_interpreter": "code_interpreter",
        "computer_use": "computer_use_preview",
    }

    def __init__(
        self,
        model: str = "gpt-4o",
        api: Literal["completions", "responses"] = "completions",
        instructions: str | None = None,
        builtin_tools: list[str] | None = None,
        auto_chain: bool = False,
        auto_chain_reasoning: bool = False,
        interceptor: BaseInterceptor[httpx.Request, httpx.Response] | None = None,
        ...
    ) -> None:
```

关键特性：

- **双 API 支持**：`api="completions"` 使用经典 Chat Completions API，`api="responses"` 使用新的 Responses API
- **内建工具**：web_search、file_search、code_interpreter、computer_use
- **Responses API 状态管理**：`auto_chain` 自动追踪 response_id 实现多轮对话
- **HTTP Interceptor 支持**：通过 `BaseInterceptor` 在传输层拦截请求/响应

### 11.4.2 AnthropicCompletion

```python
class AnthropicCompletion(BaseLLM):
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,  # Anthropic 必须指定
        thinking: AnthropicThinkingConfig | None = None,
        response_format: type[BaseModel] | None = None,
        ...
    ):
```

Anthropic Provider 的特殊处理：

- **max_tokens 必填**：Anthropic API 与 OpenAI 不同，要求显式指定 max_tokens
- **Thinking 模式**：通过 `AnthropicThinkingConfig` 控制 Claude 的思考过程，支持 `budget_tokens` 参数限制思考 token 数量
- **Stop sequences 映射**：将 BaseLLM 的 `stop` 属性映射为 Anthropic 的 `stop_sequences`
- **结构化输出**：Claude 4.5 系列模型支持原生结构化输出，旧模型通过 tool-based fallback

```python
class AnthropicThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled"]
    budget_tokens: int | None = None
```

### 11.4.3 GeminiCompletion

```python
class GeminiCompletion(BaseLLM):
    def __init__(
        self,
        model: str = "gemini-2.0-flash-001",
        project: str | None = None,       # Vertex AI 项目 ID
        location: str | None = None,       # Vertex AI 区域
        use_vertexai: bool | None = None,  # 是否使用 Vertex AI
        thinking_config: types.ThinkingConfig | None = None,
        ...
    ):
```

Gemini Provider 的双模式设计：

- **Gemini API**：使用 API Key 直接调用 Google AI
- **Vertex AI**：使用 ADC（Application Default Credentials）或 Express Mode（API Key + Vertex AI）
- **ThinkingConfig**：Gemini 2.5+ 和 3+ 系列的思考模式控制

### 11.4.4 BedrockCompletion

```python
class BedrockCompletion(BaseLLM):
    """AWS Bedrock 原生实现，使用 Converse API"""
```

Bedrock Provider 的独特之处：

- **IAM 认证**：通过 boto3 Session 和 AWS IAM 角色进行认证，不需要 API Key
- **Converse API**：使用 Bedrock 的统一 Converse API，不依赖各模型供应商的原生格式
- **Guardrail 配置**：支持 AWS Bedrock Guardrails 内容过滤
- **跨区域模型 ID**：支持 `us.anthropic.claude-3-5-sonnet-20241022-v2:0` 这样的区域前缀格式

### 11.4.5 AzureCompletion

```python
class AzureCompletion(BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
        ...
    ):
```

Azure Provider 使用 `azure-ai-inference` SDK 而非 OpenAI SDK，支持 Azure AI Model Catalog 中的所有模型。

## 11.5 同一 Crew 中的多模型路由

CrewAI 的一个核心能力是同一 Crew 中的不同 Agent 可以使用不同的模型。这得益于 LLM 接口层的设计：每个 Agent 持有独立的 LLM 实例。

```python
from crewai import Agent, LLM

# 研究员使用 Claude 的深度推理能力
researcher = Agent(
    role="Researcher",
    llm=LLM("anthropic/claude-3-7-sonnet-20250219",
            thinking={"type": "enabled", "budget_tokens": 10000}),
    ...
)

# 写手使用 GPT-4o 的快速生成能力
writer = Agent(
    role="Writer",
    llm=LLM("openai/gpt-4o"),
    ...
)

# 审核员使用性价比高的 Gemini Flash
reviewer = Agent(
    role="Reviewer",
    llm=LLM("gemini/gemini-2.0-flash"),
    ...
)
```

由于 `LLM.__new__` 的工厂路由，这三个 LLM 实例分别是 `AnthropicCompletion`、`OpenAICompletion` 和 `GeminiCompletion` 的实例。但上层代码只需要调用统一的 `call()` 方法即可。

## 11.6 统一参数管理

### 11.6.1 LiteLLM 回退的参数体系

当使用 LiteLLM 回退时，`LLM.__init__` 接管参数管理：

```python
def __init__(
    self,
    model: str,
    timeout: float | int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    n: int | None = None,
    stop: str | list[str] | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: dict[int, float] | None = None,
    response_format: type[BaseModel] | None = None,
    seed: int | None = None,
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None,
    stream: bool = False,
    interceptor: BaseInterceptor[...] | None = None,
    thinking: AnthropicThinkingConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
```

所有未被显式声明的参数通过 `**kwargs` 收集到 `self.additional_params` 中，并在调用 `litellm.completion()` 时展开：

```python
params = {
    "model": self.model,
    "messages": formatted_messages,
    "temperature": self.temperature,
    "max_tokens": self.max_tokens or self.max_completion_tokens,
    "stop": self.stop or None,
    ...
    **self.additional_params,  # Provider 特定参数在这里透传
}
# 移除 None 值
return {k: v for k, v in params.items() if v is not None}
```

### 11.6.2 Context Window 大小管理

`LLM` 维护了一个详尽的 context window 大小映射表 `LLM_CONTEXT_WINDOW_SIZES`，覆盖了 OpenAI、Gemini、Groq、SambaNova、Bedrock、Mistral 等多个 Provider 的模型：

```python
LLM_CONTEXT_WINDOW_SIZES: Final[dict[str, int]] = {
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-4.1": 1047576,
    "gemini-1.5-pro": 2097152,
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    ...
}
```

获取 context window 时使用 85% 的安全系数：

```python
CONTEXT_WINDOW_USAGE_RATIO: Final[float] = 0.85

def get_context_window_size(self) -> int:
    for key, value in LLM_CONTEXT_WINDOW_SIZES.items():
        if self.model.startswith(key):
            self.context_window_size = int(value * CONTEXT_WINDOW_USAGE_RATIO)
    return self.context_window_size
```

## 11.7 Provider 特定的消息格式化

LiteLLM 回退路径中，`_format_messages_for_provider` 方法处理各 Provider 的消息格式差异：

```python
def _format_messages_for_provider(self, messages):
    # O1 模型不支持 system 角色
    if "o1" in self.model.lower():
        for msg in messages:
            if msg["role"] == "system":
                msg["role"] = "assistant"

    # Mistral 要求最后一条消息是 user 或 tool
    if "mistral" in self.model.lower():
        if messages and messages[-1]["role"] == "assistant":
            return [*messages, {"role": "user", "content": "Please continue."}]

    # Anthropic 要求首条消息是 user 角色
    if self.is_anthropic:
        if not messages or messages[0]["role"] == "system":
            return [{"role": "user", "content": "."}, *messages]
```

## 11.8 HTTP 传输层拦截器

`BaseInterceptor` 提供了在 HTTP 传输层拦截和修改请求/响应的能力：

```python
class BaseInterceptor(ABC, Generic[T, U]):
    @abstractmethod
    def on_outbound(self, message: T) -> T:
        """拦截出站消息"""
        ...

    @abstractmethod
    def on_inbound(self, message: U) -> U:
        """拦截入站消息"""
        ...

    async def aon_outbound(self, message: T) -> T:
        raise NotImplementedError

    async def aon_inbound(self, message: U) -> U:
        raise NotImplementedError
```

各 Provider 通过 `HTTPTransport` 和 `AsyncHTTPTransport` 将 Interceptor 注入到 HTTP Client 中。这个机制允许用户在不修改 Provider 代码的情况下添加自定义 Header、记录请求日志、或者在传输层做安全审计。

## 11.9 模型常量与验证

`llms/constants.py` 定义了各 Provider 支持的模型列表，使用 `TypeAlias` 和 `Literal` 类型提供编译时的类型安全：

```python
OpenAIModels: TypeAlias = Literal[
    "gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini",
    "gpt-4.1", "gpt-4.1-mini", "o1", "o3-mini", "o4-mini",
    ...
]
OPENAI_MODELS: list[OpenAIModels] = [...]

AnthropicModels: TypeAlias = Literal[
    "claude-opus-4-5", "claude-sonnet-4-5",
    "claude-3-7-sonnet-20250219", ...
]
ANTHROPIC_MODELS: list[AnthropicModels] = [...]
```

当模型名不在常量列表中时，系统还会通过模式匹配进行推断：

```python
@classmethod
def _matches_provider_pattern(cls, model: str, provider: str) -> bool:
    model_lower = model.lower()
    if provider == "openai":
        return any(model_lower.startswith(p)
                   for p in ["gpt-", "o1", "o3", "o4", "whisper-"])
    if provider == "anthropic":
        return any(model_lower.startswith(p)
                   for p in ["claude-", "anthropic."])
    if provider == "gemini":
        return any(model_lower.startswith(p)
                   for p in ["gemini-", "gemma-", "learnlm-"])
    if provider == "bedrock":
        return "." in model_lower  # Bedrock 模型名包含点号
    ...
```

这种双重验证机制确保了既能支持已知模型的精确匹配，又能为新发布的模型提供基于命名规范的自动推断。

## 11.10 FilteredStream：静默 LiteLLM 噪音

LiteLLM 运行时会向 stdout/stderr 输出大量调试信息。`LLM` 模块通过 `FilteredStream` 代理拦截这些输出：

```python
class FilteredStream(io.TextIOBase):
    def write(self, s: str) -> int:
        lower_s = s.lower()
        if "litellm.info:" in lower_s or \
           "Consider using a smaller input" in lower_s:
            return 0
        return self._original_stream.write(s)

if not isinstance(sys.stdout, FilteredStream):
    sys.stdout = FilteredStream(sys.stdout)
if not isinstance(sys.stderr, FilteredStream):
    sys.stderr = FilteredStream(sys.stderr)
```

这个全局过滤器在模块加载时自动安装，带有重复包装保护。

## 本章要点

- **BaseLLM** 是所有 LLM 实现的抽象基类，定义了 `call` / `acall` 核心接口以及事件发射、token 追踪、stop words 处理等模板方法
- **LLM.__new__** 实现了工厂模式的路由逻辑：先尝试匹配原生 Provider（OpenAI / Anthropic / Gemini / Bedrock / Azure），失败后回退到 LiteLLM
- **五大原生 Provider** 各自封装了对应 SDK 的特有能力：OpenAI 的 Responses API、Anthropic 的 Thinking 模式、Gemini 的 Vertex AI 双模式、Bedrock 的 IAM 认证和 Converse API、Azure 的 AI Inference SDK
- **模型路由**支持三种指定方式：显式 provider 参数、`provider/model` 斜杠格式、纯模型名自动推断
- **统一参数管理**通过 `additional_params` 透传 Provider 特定参数，context window 大小使用 85% 安全系数
- **BaseInterceptor** 提供了传输层的请求/响应拦截能力，可用于审计、日志、自定义 Header 等场景
- **Token 使用量追踪**在 BaseLLM 层面统一实现，通过 fallback 链兼容各 Provider 不同的字段命名
- 同一 Crew 中的不同 Agent 可以使用不同的 LLM 实例，实现灵活的多模型协作
