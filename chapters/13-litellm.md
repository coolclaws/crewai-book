# 第 13 章　LiteLLM 集成与模型路由

前面两章分析了 CrewAI 的统一 LLM 接口和 Hook 系统。本章将深入探讨 CrewAI 如何集成 LiteLLM 作为后备模型路由层，以及这一集成如何使框架获得了对 100+ 模型供应商的即时支持。

## 13.1 LiteLLM 在 CrewAI 中的定位

CrewAI 的 LLM 层采用了"原生 Provider 优先，LiteLLM 兜底"的双层架构：

```
用户代码: LLM("deepseek/deepseek-chat")
                    │
                    ▼
            LLM.__new__()
            ┌───────────────┐
            │ 1. 检查显式     │
            │    provider    │
            │ 2. 解析 / 前缀  │
            │ 3. 查找常量表   │
            │ 4. 模式匹配     │
            └───────┬───────┘
                    │
         ┌──────────┴──────────┐
         │ 匹配原生 Provider？ │
         ├──── YES ────┐       │
         │             ▼       │
         │   OpenAICompletion  │
         │   AnthropicCompl.   │
         │   GeminiCompletion  │
         │   BedrockCompletion │
         │   AzureCompletion   │
         │                     │
         ├──── NO ─────────────┤
         │                     ▼
         │              LLM (LiteLLM)
         │              is_litellm=True
         └─────────────────────┘
```

LiteLLM 作为可选依赖安装：

```bash
# 仅安装原生 Provider
pip install crewai

# 安装 LiteLLM 支持
pip install 'crewai[litellm]'
```

当模型不匹配任何原生 Provider 且 LiteLLM 未安装时，CrewAI 会给出明确的错误提示：

```python
if not LITELLM_AVAILABLE:
    error_msg = (
        f"Unable to initialize LLM with model '{model}'. "
        f"The model did not match any supported native provider "
        f"({native_list}), and the LiteLLM fallback package is not installed.\n\n"
        f"To fix this, either:\n"
        f"  1. Install LiteLLM for broad model support: uv add 'crewai[litellm]'\n"
        f"or\n"
        f"pip install litellm\n"
    )
    raise ImportError(error_msg)
```

## 13.2 模型字符串约定

CrewAI 延续了 LiteLLM 的模型命名约定，使用 `provider/model-name` 格式：

```python
# 原生 Provider（直接 SDK 调用）
LLM("gpt-4o")                              # OpenAI（自动推断）
LLM("claude-3-5-sonnet-20241022")           # Anthropic（自动推断）
LLM("openai/gpt-4o")                       # OpenAI（显式前缀）
LLM("anthropic/claude-3-5-sonnet-20241022") # Anthropic（显式前缀）
LLM("gemini/gemini-2.0-flash")             # Gemini（显式前缀）
LLM("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")  # AWS Bedrock

# LiteLLM 回退（通过 litellm.completion 调用）
LLM("deepseek/deepseek-chat")              # DeepSeek
LLM("groq/llama-3.3-70b-versatile")        # Groq
LLM("ollama/llama3.2")                     # Ollama 本地模型
LLM("openrouter/deepseek/deepseek-chat")   # OpenRouter
LLM("together_ai/meta-llama/Llama-3-70b")  # Together AI
LLM("sambanova/Meta-Llama-3.3-70B-Instruct") # SambaNova
```

### 13.2.1 路由决策流程

当模型字符串包含 `/` 时，`LLM.__new__` 的处理逻辑是：

```python
elif "/" in model:
    prefix, _, model_part = model.partition("/")

    provider_mapping = {
        "openai": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "azure": "azure",
        "azure_openai": "azure",
        "google": "gemini",
        "gemini": "gemini",
        "bedrock": "bedrock",
        "aws": "bedrock",
    }

    canonical_provider = provider_mapping.get(prefix.lower())

    if canonical_provider and cls._validate_model_in_constants(
        model_part, canonical_provider
    ):
        # 匹配原生 Provider -> 使用原生 SDK
        provider = canonical_provider
        use_native = True
        model_string = model_part
    else:
        # 不匹配 -> 回退到 LiteLLM
        provider = prefix
        use_native = False
        model_string = model_part
```

关键在于 `_validate_model_in_constants` 方法的双重验证：

1. **常量表精确匹配**：检查模型名是否在 `OPENAI_MODELS`、`ANTHROPIC_MODELS` 等列表中
2. **模式匹配兜底**：如果常量表未命中，通过前缀模式判断（如 `gpt-` 开头推断为 OpenAI）

这意味着即使是 OpenAI 新发布的模型（尚未加入常量表），只要名称以 `gpt-` 或 `o1` 等已知前缀开头，仍然会路由到原生 Provider。

### 13.2.2 不含 `/` 的纯模型名

当模型字符串不含 `/` 时，系统通过 `_infer_provider_from_model` 在所有 Provider 的常量表中查找：

```python
@classmethod
def _infer_provider_from_model(cls, model: str) -> str:
    if model in OPENAI_MODELS:
        return "openai"
    if model in ANTHROPIC_MODELS:
        return "anthropic"
    if model in GEMINI_MODELS:
        return "gemini"
    if model in BEDROCK_MODELS:
        return "bedrock"
    if model in AZURE_MODELS:
        return "azure"
    return "openai"  # 默认回退
```

注意默认回退到 `"openai"` 而非 LiteLLM，这是因为大多数 OpenAI 兼容的第三方服务都可以通过 OpenAI SDK 直接调用。

## 13.3 LiteLLM 集成的核心实现

### 13.3.1 条件导入

CrewAI 对 LiteLLM 的导入做了完善的错误处理：

```python
try:
    import litellm
    from litellm.exceptions import ContextWindowExceededError
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.get_supported_openai_params import (
        get_supported_openai_params,
    )
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall, Choices, Function, ModelResponse,
    )
    from litellm.utils import supports_response_schema

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None
    Choices = None
    ContextWindowExceededError = Exception
    ...
```

当 LiteLLM 不可用时，所有类型引用都被替换为 `None` 或通用 `Exception`，确保模块仍然可以导入——只是在实际创建 LiteLLM 实例时才会报错。

### 13.3.2 全局配置

LiteLLM 加载后立即进行全局配置：

```python
load_dotenv()
logger = logging.getLogger(__name__)
if LITELLM_AVAILABLE:
    litellm.suppress_debug_info = True  # 抑制 LiteLLM 的调试输出
    litellm.drop_params = True          # 自动丢弃不支持的参数
```

`drop_params = True` 是一个关键配置：它让 LiteLLM 自动忽略目标 Provider 不支持的参数（如给不支持 `logprobs` 的模型传入 `logprobs` 参数时不报错），极大地简化了跨 Provider 的参数兼容性问题。

### 13.3.3 参数组装

`_prepare_completion_params` 方法将 LLM 实例的所有配置组装为 `litellm.completion()` 调用的参数：

```python
def _prepare_completion_params(self, messages, tools=None, skip_file_processing=False):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not skip_file_processing:
        messages = self._process_message_files(messages)
    formatted_messages = self._format_messages_for_provider(messages)

    params = {
        "model": self.model,
        "messages": formatted_messages,
        "timeout": self.timeout,
        "temperature": self.temperature,
        "top_p": self.top_p,
        "n": self.n,
        "stop": self.stop or None,
        "max_tokens": self.max_tokens or self.max_completion_tokens,
        "presence_penalty": self.presence_penalty,
        "frequency_penalty": self.frequency_penalty,
        "logit_bias": self.logit_bias,
        "response_format": self.response_format,
        "seed": self.seed,
        "logprobs": self.logprobs,
        "top_logprobs": self.top_logprobs,
        "api_base": self.api_base,
        "base_url": self.base_url,
        "api_version": self.api_version,
        "api_key": self.api_key,
        "stream": self.stream,
        "tools": tools,
        "reasoning_effort": self.reasoning_effort,
        **self.additional_params,
    }
    return {k: v for k, v in params.items() if v is not None}
```

最后一行过滤掉所有 `None` 值——这是与 LiteLLM 交互的关键技巧。LiteLLM 对 `None` 和**未传入**有不同的处理逻辑，过滤 `None` 值确保只传递用户明确设置的参数。

### 13.3.4 同步调用流程

```python
def call(self, messages, tools=None, callbacks=None, available_functions=None,
         from_task=None, from_agent=None, response_model=None):
    with llm_call_context() as call_id:
        # 1. 事件发射
        crewai_event_bus.emit(self, event=LLMCallStartedEvent(...))

        # 2. 参数验证（检查 response_format 兼容性）
        self._validate_call_params()

        # 3. Hook 前置检查
        if not self._invoke_before_llm_call_hooks(messages, from_agent):
            raise ValueError("LLM call blocked by before_llm_call hook")

        # 4. 组装参数
        params = self._prepare_completion_params(messages, tools)

        # 5. 分流：streaming vs non-streaming
        if self.stream:
            result = self._handle_streaming_response(params, ...)
        else:
            result = self._handle_non_streaming_response(params, ...)

        # 6. Hook 后置处理
        if isinstance(result, str):
            result = self._invoke_after_llm_call_hooks(messages, result, from_agent)

        return result
```

### 13.3.5 异步调用流程

异步路径几乎与同步路径对称，但调用 `litellm.acompletion()` 代替 `litellm.completion()`：

```python
async def acall(self, messages, tools=None, ...):
    with llm_call_context() as call_id:
        # 异步文件处理
        messages = await self._aprocess_message_files(messages)

        params = self._prepare_completion_params(messages, tools, skip_file_processing=True)

        if self.stream:
            return await self._ahandle_streaming_response(params, ...)
        return await self._ahandle_non_streaming_response(params, ...)
```

注意异步路径先调用 `_aprocess_message_files` 处理文件，然后在 `_prepare_completion_params` 中通过 `skip_file_processing=True` 跳过重复处理。

## 13.4 Streaming 实现

LiteLLM 回退路径的 streaming 实现比较复杂，因为要处理多种 chunk 格式：

```python
def _handle_streaming_response(self, params, ...):
    full_response = ""
    accumulated_tool_args = defaultdict(AccumulatedToolArgs)

    params["stream"] = True
    params["stream_options"] = {"include_usage": True}

    for chunk in litellm.completion(**params):
        # 从 chunk 中提取内容（兼容 dict 和 object 两种格式）
        choices = None
        if isinstance(chunk, dict) and "choices" in chunk:
            choices = chunk["choices"]
        elif hasattr(chunk, "choices"):
            if not isinstance(chunk.choices, type):
                choices = chunk.choices

        if choices and len(choices) > 0:
            delta = choices[0].delta if hasattr(choices[0], "delta") else None
            if delta:
                chunk_content = getattr(delta, "content", None)

                # 处理 streaming tool calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    ...  # 累积工具调用参数

        if chunk_content:
            full_response += chunk_content
            crewai_event_bus.emit(self, event=LLMStreamChunkEvent(...))

    # 提取 usage 信息
    if usage_info:
        self._track_token_usage_internal(usage_info)

    return full_response
```

`AccumulatedToolArgs` 数据模型用于在 streaming 过程中逐步收集工具调用参数：

```python
class FunctionArgs(BaseModel):
    name: str = ""
    arguments: str = ""

class AccumulatedToolArgs(BaseModel):
    function: FunctionArgs = Field(default_factory=FunctionArgs)
```

当 function name 和 arguments 都收集完毕且 arguments 是合法 JSON 时，立即执行工具调用：

```python
if current_tool_accumulator.function.name and \
   current_tool_accumulator.function.arguments and \
   available_functions:
    try:
        json.loads(current_tool_accumulator.function.arguments)
        return self._handle_tool_call([current_tool_accumulator], available_functions)
    except json.JSONDecodeError:
        continue  # 继续收集
```

## 13.5 结构化输出

CrewAI 通过 `InternalInstructor` 工具类在 LiteLLM 路径上实现结构化输出：

```python
if response_model and self.is_litellm:
    from crewai.utilities.internal_instructor import InternalInstructor

    combined_content = "\n\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in messages
    )

    instructor_instance = InternalInstructor(
        content=combined_content,
        model=response_model,
        llm=self,
    )
    result = instructor_instance.to_pydantic()
    return result.model_dump_json()
```

这里 `InternalInstructor` 将对话内容和 Pydantic 模型 schema 合并为 prompt，引导 LLM 生成符合 schema 的 JSON 输出。

## 13.6 Token 使用量追踪与 UsageMetrics

### 13.6.1 UsageMetrics 数据模型

CrewAI 在 `types/usage_metrics.py` 中定义了标准化的使用量指标：

```python
class UsageMetrics(BaseModel):
    total_tokens: int = Field(default=0, description="Total number of tokens used.")
    prompt_tokens: int = Field(default=0, description="Number of tokens used in prompts.")
    cached_prompt_tokens: int = Field(default=0, description="Number of cached prompt tokens used.")
    completion_tokens: int = Field(default=0, description="Number of tokens used in completions.")
    successful_requests: int = Field(default=0, description="Number of successful requests made.")

    def add_usage_metrics(self, usage_metrics: Self) -> None:
        self.total_tokens += usage_metrics.total_tokens
        self.prompt_tokens += usage_metrics.prompt_tokens
        self.cached_prompt_tokens += usage_metrics.cached_prompt_tokens
        self.completion_tokens += usage_metrics.completion_tokens
        self.successful_requests += usage_metrics.successful_requests
```

`add_usage_metrics` 方法支持指标的累加合并，用于将 Agent 级别的指标汇总到 Crew 级别。

### 13.6.2 多 Provider 字段兼容

不同 Provider 返回的 token 使用量字段名称不同：

| Provider | Prompt Tokens | Completion Tokens | Cached Tokens |
|----------|--------------|-------------------|---------------|
| OpenAI | `prompt_tokens` | `completion_tokens` | `cached_tokens` |
| Anthropic | `input_tokens` | `output_tokens` | - |
| Gemini | `prompt_token_count` | `candidates_token_count` | - |

`BaseLLM._track_token_usage_internal` 通过 fallback 链统一处理：

```python
def _track_token_usage_internal(self, usage_data):
    prompt_tokens = (
        usage_data.get("prompt_tokens")
        or usage_data.get("prompt_token_count")
        or usage_data.get("input_tokens")
        or 0
    )
    completion_tokens = (
        usage_data.get("completion_tokens")
        or usage_data.get("candidates_token_count")
        or usage_data.get("output_tokens")
        or 0
    )
    ...
```

### 13.6.3 LiteLLM 路径的 Usage 追踪

在 LiteLLM 路径中，非 streaming 响应的 usage 直接从 response 对象获取：

```python
response = litellm.completion(**params)

if hasattr(response, "usage") and not isinstance(response.usage, type) and response.usage:
    usage_info = response.usage
    self._track_token_usage_internal(usage_info)
```

Streaming 响应需要从最后一个 chunk 中提取 usage（通过 `stream_options={"include_usage": True}` 请求）：

```python
params["stream_options"] = {"include_usage": True}

for chunk in litellm.completion(**params):
    if hasattr(chunk, "usage") and chunk.usage is not None:
        usage_info = chunk.usage
    ...

if usage_info:
    self._track_token_usage_internal(usage_info)
```

### 13.6.4 Cost 追踪

LLM 类还维护了一个 `completion_cost` 属性，但这主要依赖 LiteLLM 的 cost tracking 能力：

```python
class LLM(BaseLLM):
    completion_cost: float | None = None
```

## 13.7 Callback 系统集成

### 13.7.1 LiteLLM Callbacks

LiteLLM 有自己的 callback 系统，CrewAI 通过 `set_callbacks` 和 `set_env_callbacks` 进行集成：

```python
@staticmethod
def set_callbacks(callbacks: list[Any]) -> None:
    """保持 litellm 中唯一的 callback 集合"""
    callback_types = [type(callback) for callback in callbacks]
    # 移除同类型的旧 callback
    for callback in litellm.success_callback[:]:
        if type(callback) in callback_types:
            litellm.success_callback.remove(callback)
    for callback in litellm._async_success_callback[:]:
        if type(callback) in callback_types:
            litellm._async_success_callback.remove(callback)
    litellm.callbacks = callbacks
```

### 13.7.2 环境变量配置 Callbacks

支持通过环境变量配置 LiteLLM 的 callback：

```python
@staticmethod
def set_env_callbacks() -> None:
    success_callbacks_str = os.environ.get("LITELLM_SUCCESS_CALLBACKS", "")
    if success_callbacks_str:
        success_callbacks = [cb.strip() for cb in success_callbacks_str.split(",")]

    failure_callbacks_str = os.environ.get("LITELLM_FAILURE_CALLBACKS", "")
    if failure_callbacks_str:
        failure_callbacks = [cb.strip() for cb in failure_callbacks_str.split(",")]
        litellm.success_callback = success_callbacks
        litellm.failure_callback = failure_callbacks
```

这允许通过环境变量接入 Langfuse、LangSmith 等可观测性平台：

```bash
export LITELLM_SUCCESS_CALLBACKS="langfuse,langsmith"
export LITELLM_FAILURE_CALLBACKS="langfuse"
```

## 13.8 Provider 特定的消息格式化

LiteLLM 回退路径中需要处理各 Provider 的消息格式差异。`_format_messages_for_provider` 处理了多种边界情况：

```python
def _format_messages_for_provider(self, messages):
    # O1 模型：不支持 system 角色，转为 assistant
    if "o1" in self.model.lower():
        for msg in messages:
            if msg["role"] == "system":
                msg["role"] = "assistant"

    # Mistral：最后一条消息必须是 user 或 tool
    if "mistral" in self.model.lower():
        if messages and messages[-1]["role"] == "assistant":
            return [*messages, {"role": "user", "content": "Please continue."}]

    # Ollama：不支持最后一条消息是 assistant
    if "ollama" in self.model.lower():
        if messages and messages[-1]["role"] == "assistant":
            return [*messages, {"role": "user", "content": ""}]

    # Anthropic：第一条消息必须是 user
    if self.is_anthropic:
        if not messages or messages[0]["role"] == "system":
            return [{"role": "user", "content": "."}, *messages]
```

## 13.9 Unsupported Parameter 自动重试

LiteLLM 路径有一个优雅的自动重试机制：当 Provider 不支持 `stop` 参数时，自动丢弃该参数并重试：

```python
except Exception as e:
    unsupported_stop = "Unsupported parameter" in str(e) and "'stop'" in str(e)

    if unsupported_stop:
        if "additional_drop_params" in self.additional_params:
            self.additional_params["additional_drop_params"].append("stop")
        else:
            self.additional_params = {"additional_drop_params": ["stop"]}

        logging.info("Retrying LLM call without the unsupported 'stop'")
        return self.call(messages, tools=tools, ...)
```

这利用了 LiteLLM 的 `additional_drop_params` 机制，在重试时告诉 LiteLLM 自动丢弃 `stop` 参数。

## 13.10 Context Window 管理

### 13.10.1 预定义大小表

`LLM` 类维护了一个全面的 context window 大小映射表，覆盖主流 Provider：

```python
LLM_CONTEXT_WINDOW_SIZES: Final[dict[str, int]] = {
    # OpenAI
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-4.1": 1047576,

    # Gemini
    "gemini-1.5-pro": 2097152,
    "gemini-2.0-flash": 1048576,

    # Bedrock
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "amazon.nova-pro-v1:0": 300000,

    # Groq
    "llama-3.3-70b-versatile": 128000,

    # DeepSeek
    "deepseek-chat": 128000,
    ...
}
```

### 13.10.2 安全系数

获取 context window 时使用 85% 的安全系数，为 system prompt、tool definitions 等留出余量：

```python
CONTEXT_WINDOW_USAGE_RATIO: Final[float] = 0.85

def get_context_window_size(self) -> int:
    if self.context_window_size != 0:
        return self.context_window_size  # 缓存

    self.context_window_size = int(DEFAULT_CONTEXT_WINDOW_SIZE * CONTEXT_WINDOW_USAGE_RATIO)
    for key, value in LLM_CONTEXT_WINDOW_SIZES.items():
        if self.model.startswith(key):
            self.context_window_size = int(value * CONTEXT_WINDOW_USAGE_RATIO)
    return self.context_window_size
```

### 13.10.3 Context Window 溢出处理

当 LLM 调用触发 context window 溢出时，CrewAI 将 LiteLLM 的异常转换为自己的异常类型：

```python
try:
    response = litellm.completion(**params)
except ContextWindowExceededError as e:
    raise LLMContextLengthExceededError(str(e)) from e
```

`LLMContextLengthExceededError` 被 `CrewAgentExecutor._invoke_loop` 捕获，根据 `respect_context_window` 配置决定是进行内容摘要还是终止执行。

## 13.11 Function Calling 支持检测

LiteLLM 回退路径通过 LiteLLM 的能力检测 API 判断模型是否支持 function calling：

```python
def supports_function_calling(self) -> bool:
    try:
        provider = self._get_custom_llm_provider()
        return litellm.utils.supports_function_calling(
            self.model, custom_llm_provider=provider
        )
    except Exception:
        return False

def supports_stop_words(self) -> bool:
    try:
        params = get_supported_openai_params(model=self.model)
        return params is not None and "stop" in params
    except Exception:
        return False
```

## 13.12 第三方模型集成

`llms/third_party/` 目录预留了第三方 LLM 实现的扩展点。目前该目录只有一个空的 `__init__.py`：

```python
"""Third-party LLM implementations for crewAI."""
```

用户可以通过继承 `BaseLLM` 来创建自定义的 LLM 实现：

```python
from crewai.llms.base_llm import BaseLLM

class MyCustomLLM(BaseLLM):
    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        # 初始化自定义客户端

    def call(self, messages, tools=None, callbacks=None,
             available_functions=None, from_task=None,
             from_agent=None, response_model=None):
        # 实现自定义调用逻辑
        formatted = self._format_messages(messages)
        self._emit_call_started_event(messages, tools, ...)

        # 调用自定义 API
        response = my_api_call(formatted)

        # 追踪 token 使用量
        self._track_token_usage_internal(response.usage)
        self._emit_call_completed_event(response.text, LLMCallType.LLM_CALL, ...)

        return response.text
```

自定义 LLM 可以直接传给 Agent：

```python
agent = Agent(
    role="Researcher",
    llm=MyCustomLLM(model="my-model", api_key="..."),
)
```

## 13.13 Multimodal 支持检测

LiteLLM 回退路径通过模型名前缀检测 multimodal 支持：

```python
def supports_multimodal(self) -> bool:
    vision_prefixes = (
        "gpt-4o", "gpt-4-turbo", "gpt-4-vision", "gpt-4.1",
        "claude-3", "claude-4", "gemini",
    )
    model_lower = self.model.lower()
    return any(
        model_lower.startswith(p) or f"/{p}" in model_lower
        for p in vision_prefixes
    )
```

注意检查条件同时覆盖了直接模型名（`gpt-4o-...`）和带前缀的模型名（`openai/gpt-4o-...`）。

## 13.14 实例复制

LLM 实例支持浅拷贝和深拷贝，这在 Crew 的并行执行中很重要：

```python
def __copy__(self) -> LLM:
    filtered_params = {
        k: v for k, v in self.additional_params.items()
        if k not in ["model", "is_litellm", "temperature", ...]
    }
    return LLM(
        model=self.model,
        is_litellm=self.is_litellm,
        temperature=self.temperature,
        ...
        **filtered_params,
    )
```

`__copy__` 和 `__deepcopy__` 都通过创建新的 `LLM` 实例来实现，确保复制后的实例是完全独立的（包括独立的 token 使用量追踪）。

## 本章要点

- CrewAI 采用"原生 Provider 优先，LiteLLM 兜底"的双层路由架构，LiteLLM 作为可选依赖提供 100+ Provider 支持
- 模型字符串遵循 `provider/model-name` 约定，`LLM.__new__` 通过常量表精确匹配和模式匹配双重验证决定路由
- LiteLLM 集成的核心是 `litellm.completion()` 和 `litellm.acompletion()`，通过 `drop_params=True` 自动兼容参数差异
- `_prepare_completion_params` 将所有 LLM 配置组装为参数字典，过滤 `None` 值确保仅传递显式设置的参数
- Streaming 实现使用 `AccumulatedToolArgs` 逐步收集工具调用参数，在 JSON 完整时立即执行
- `UsageMetrics` 提供标准化的 token 使用量追踪，通过 fallback 链兼容各 Provider 不同的字段命名
- 环境变量 `LITELLM_SUCCESS_CALLBACKS` / `LITELLM_FAILURE_CALLBACKS` 支持接入 Langfuse、LangSmith 等可观测性平台
- Context window 管理使用 85% 安全系数，溢出时转换为 `LLMContextLengthExceededError` 由上层 executor 处理
- 不支持的参数（如 `stop`）触发自动重试机制，利用 LiteLLM 的 `additional_drop_params` 优雅降级
- `BaseLLM` 的开放设计允许用户通过继承实现自定义 LLM Provider，直接传给 Agent 使用
