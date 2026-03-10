# 第 21 章　CLI：创建、训练与部署

CrewAI 提供了一套完整的命令行工具（CLI），让开发者可以从项目脚手架搭建、本地运行、训练调优到云端部署，全部在终端中完成。CLI 基于 Click 框架构建，采用分层命令组（command group）设计，每个子命令对应一个独立的 Python 模块。本章将深入 `cli/` 目录的源码，逐一剖析每条命令的实现机制。

## 21.1 整体架构：Click 命令组

CLI 的入口在 `cli/cli.py`，通过 `@click.group()` 定义了顶层命令组 `crewai`：

```python
@click.group()
@click.version_option(get_version("crewai"))
def crewai():
    """Top-level command group for crewai."""
```

所有子命令都通过装饰器挂载到这个组下。Click 框架的核心概念是：

- **`@click.group()`** — 定义一个命令组，可包含多个子命令
- **`@crewai.command()`** — 向组中添加一个命令
- **`@crewai.group()`** — 在组内嵌套另一个命令组（如 `deploy`、`flow`、`tool`）
- **`@click.option()`** / **`@click.argument()`** — 定义命令的参数和选项

顶层命令结构如下：

| 命令 | 类型 | 说明 |
|------|------|------|
| `crewai create` | command | 创建 Crew 或 Flow 项目 |
| `crewai run` | command | 运行 Crew（或 Flow） |
| `crewai train` | command | 训练 Crew |
| `crewai test` | command | 测试与评估 Crew |
| `crewai replay` | command | 从指定 Task 重放 |
| `crewai chat` | command | 交互式对话模式 |
| `crewai deploy` | group | 部署相关命令组 |
| `crewai flow` | group | Flow 相关命令组 |
| `crewai tool` | group | 工具仓库命令组 |
| `crewai memory` | command | Memory TUI 浏览器 |
| `crewai reset-memories` | command | 重置记忆存储 |
| `crewai install` | command | 安装 Crew 依赖 |
| `crewai version` | command | 显示版本信息 |
| `crewai login` | command | 登录 CrewAI+ 平台 |
| `crewai triggers` | group | Trigger 相关命令组 |
| `crewai org` | group | 组织管理命令组 |
| `crewai config` | group | CLI 配置命令组 |
| `crewai traces` | group | Trace 采集管理 |
| `crewai env` | group | 环境变量查看 |
| `crewai uv` | command | uv 包管理器封装 |

## 21.2 crewai create：项目脚手架

`create` 命令是最常用的入口之一，支持创建两种类型的项目：

```python
@crewai.command()
@click.argument("type", type=click.Choice(["crew", "flow"]))
@click.argument("name")
@click.option("--provider", type=str, help="The provider to use for the crew")
@click.option("--skip_provider", is_flag=True, help="Skip provider validation")
def create(type, name, provider, skip_provider=False):
    """Create a new crew, or flow."""
    if type == "crew":
        create_crew(name, provider, skip_provider)
    elif type == "flow":
        create_flow(name)
```

### 21.2.1 创建 Crew 项目

`create_crew()` 在 `cli/create_crew.py` 中实现，核心流程分三步：

**第一步：创建目录结构**

`create_folder_structure()` 函数负责将用户提供的项目名转换为合法的 Python 模块名，并创建标准目录：

```python
def create_folder_structure(name, parent_folder=None):
    folder_name = name.replace(" ", "_").replace("-", "_").lower()
    class_name = name.replace("_", " ").replace("-", " ").title().replace(" ", "")

    folder_path.mkdir(parents=True)
    (folder_path / "tests").mkdir(exist_ok=True)
    (folder_path / "knowledge").mkdir(exist_ok=True)
    if not parent_folder:
        (folder_path / "src" / folder_name).mkdir(parents=True)
        (folder_path / "src" / folder_name / "tools").mkdir(parents=True)
        (folder_path / "src" / folder_name / "config").mkdir(parents=True)
```

生成的目录结构为：

```
my_crew/
  AGENTS.md
  pyproject.toml
  README.md
  .gitignore
  knowledge/
    user_preference.txt
  tests/
  src/
    my_crew/
      __init__.py
      main.py
      crew.py
      config/
        agents.yaml
        tasks.yaml
      tools/
        __init__.py
        custom_tool.py
```

函数中包含了多项验证：
- 名称不能为空
- 不能以数字开头（不合法的 Python 标识符）
- 不能是 Python 关键字
- 不能与 pyproject.toml 中的保留脚本名冲突

**第二步：Provider 选择**

如果没有指定 `--skip_provider`，CLI 会交互式引导用户选择 LLM Provider（如 OpenAI、Anthropic、Google 等），并选择具体模型。选择结果写入 `.env` 文件：

```python
selected_provider = select_provider(provider_models)
selected_model = select_model(selected_provider, provider_models)
env_vars["MODEL"] = selected_model
write_env_file(folder_path, env_vars)
```

**第三步：模板文件复制**

<span v-pre>从 `cli/templates/crew/` 目录复制模板文件，并用 `{{name}}`、`{{crew_name}}`、`{{folder_name}}` 等占位符进行替换。关键模板文件包括：</span>

- **pyproject.toml** — 定义项目元数据和脚本入口
- **crew.py** — `@CrewBase` 装饰的 Crew 类
- **main.py** — 包含 `run()`、`train()`、`replay()`、`test()` 入口函数
- **agents.yaml** / **tasks.yaml** — Agent 和 Task 的 YAML 配置

### 21.2.2 pyproject.toml 中的脚本入口

模板中的 `pyproject.toml` 定义了关键的脚本映射：

```toml v-pre
[project.scripts]
{{folder_name}} = "{{folder_name}}.main:run"
run_crew = "{{folder_name}}.main:run"
train = "{{folder_name}}.main:train"
replay = "{{folder_name}}.main:replay"
test = "{{folder_name}}.main:test"
run_with_trigger = "{{folder_name}}.main:run_with_trigger"

[tool.crewai]
type = "crew"
```

`[tool.crewai]` 段标识项目类型为 `crew` 或 `flow`，这在 `crewai run` 时用于自动判别。

### 21.2.3 创建 Flow 项目

`create_flow()` 在 `cli/create_flow.py` 中实现，生成的目录结构与 Crew 项目类似但增加了 `crews/` 子目录：

```
my_flow/
  pyproject.toml
  .gitignore
  .env
  src/
    my_flow/
      __init__.py
      main.py
      tools/
      crews/
        poem_crew/
          ...
```

<span v-pre>Flow 项目内置了一个示例 `poem_crew`，展示了 Flow 如何组合多个 Crew。模板文件中的占位符 `{{flow_name}}` 会被替换为 CamelCase 的类名。</span>

## 21.3 crewai run：运行 Crew

`run` 命令在 `cli/run_crew.py` 中实现：

```python
@crewai.command()
def run():
    """Run the Crew."""
    run_crew()
```

`run_crew()` 的核心逻辑是通过 `uv run` 执行项目中定义的脚本入口：

```python
def run_crew() -> None:
    pyproject_data = read_toml()
    is_flow = pyproject_data.get("tool", {}).get("crewai", {}).get("type") == "flow"
    crew_type = CrewType.FLOW if is_flow else CrewType.STANDARD
    execute_command(crew_type)

def execute_command(crew_type: CrewType) -> None:
    command = ["uv", "run", "kickoff" if crew_type == CrewType.FLOW else "run_crew"]
```

关键设计点：

1. **自动检测项目类型** — 读取 `pyproject.toml` 中 `[tool.crewai].type` 字段
2. **统一入口** — `crewai run` 同时支持 Crew 和 Flow，不需要分别记忆不同命令
3. **uv 集成** — 使用 `uv run` 确保在正确的虚拟环境中执行
4. **工具仓库凭证注入** — 通过 `build_env_with_tool_repository_credentials()` 将 tool 仓库的认证信息注入环境变量

## 21.4 crewai train：训练 Crew

训练命令让用户通过多次迭代来优化 Agent 的行为：

```python
@crewai.command()
@click.option("-n", "--n_iterations", type=int, default=5,
    help="Number of iterations to train the crew")
@click.option("-f", "--filename", type=str,
    default="trained_agents_data.pkl",
    help="Path to a custom file for training")
def train(n_iterations: int, filename: str):
    """Train the crew."""
    train_crew(n_iterations, filename)
```

底层调用 `uv run train <n_iterations> <filename>`，对应模板 `main.py` 中的 `train()` 函数：

```python v-pre
def train():
    inputs = {"topic": "AI LLMs", "current_year": str(datetime.now().year)}
    {{crew_name}}().crew().train(
        n_iterations=int(sys.argv[1]),
        filename=sys.argv[2],
        inputs=inputs
    )
```

训练数据保存为 `.pkl` 文件，包含多轮迭代中 Agent 的最优表现数据。`train_crew()` 中包含输入验证：

```python
if n_iterations <= 0:
    raise ValueError("The number of iterations must be a positive integer.")
if not filename.endswith(".pkl"):
    raise ValueError("The filename must not end with .pkl")
```

## 21.5 crewai test：测试与评估

`test` 命令用于自动化评估 Crew 的输出质量：

```python
@crewai.command()
@click.option("-n", "--n_iterations", type=int, default=3,
    help="Number of iterations to Test the crew")
@click.option("-m", "--model", type=str, default="gpt-4o-mini",
    help="LLM Model to run the tests on the Crew.")
def test(n_iterations: int, model: str):
    """Test the crew and evaluate the results."""
    evaluate_crew(n_iterations, model)
```

底层调用 `uv run test <n_iterations> <model>`，最终调用 `Crew.test()` 方法。评估使用指定的 LLM（默认 `gpt-4o-mini`）对 Crew 输出进行打分。

### 21.5.1 实验框架（experimental/evaluation）

CrewAI 还提供了更高级的实验框架用于系统化测试。`ExperimentRunner` 接受一个 dataset（测试用例列表），每个用例包含：

```python
dataset = [
    {
        "inputs": {"topic": "AI Safety"},
        "expected_score": 0.7,
        "identifier": "test_ai_safety"
    }
]
```

运行流程：

```python
class ExperimentRunner:
    def run(self, crew=None, agents=None, print_summary=False):
        for test_case in self.dataset:
            result = self._run_test_case(test_case, crew=crew, agents=agents)
            results.append(result)
        return ExperimentResults(results)
```

每个测试用例执行完后，通过 `AgentEvaluator` 评估多个维度的指标（goal metrics、reasoning metrics、semantic quality metrics、tools metrics），然后与期望分数比较。

`testing.py` 模块提供了便捷的断言函数：

```python
def run_experiment(dataset, crew=None, agents=None, verbose=False):
    runner = ExperimentRunner(dataset=dataset)
    return runner.run(agents=agents, crew=crew, print_summary=verbose)

def assert_experiment_successfully(experiment_results, baseline_filepath=None):
    # 检查失败的测试用例
    # 与基线比较，检测回归
```

## 21.6 crewai replay：任务重放

`replay` 命令允许从指定的 Task 开始重新执行 Crew：

```python
@crewai.command()
@click.option("-t", "--task_id", type=str,
    help="Replay the crew from this task ID, including all subsequent tasks.")
def replay(task_id: str) -> None:
    """Replay the crew execution from a specific task."""
    replay_task_command(task_id)
```

实现非常简洁，底层调用 `uv run replay <task_id>`：

```python
def replay_task_command(task_id: str) -> None:
    command = ["uv", "run", "replay", task_id]
    subprocess.run(command, capture_output=False, text=True, check=True)
```

这对应模板中的 `replay()` 函数：

```python v-pre
def replay():
    {{crew_name}}().crew().replay(task_id=sys.argv[1])
```

`Crew.replay()` 从 `KickoffTaskOutputsSQLiteStorage` 中读取之前执行的任务输出，跳过已完成的任务，从指定 Task 开始重新执行。这在调试长 Pipeline 时非常有用——你不需要重新执行所有前置任务。

### 21.6.1 查看任务输出日志

配合 `replay` 使用的还有 `log-tasks-outputs` 命令：

```python
@crewai.command()
def log_tasks_outputs() -> None:
    """Retrieve your latest crew.kickoff() task outputs."""
    storage = KickoffTaskOutputsSQLiteStorage()
    tasks = storage.load()
    for index, task in enumerate(tasks, 1):
        click.echo(f"Task {index}: {task['task_id']}")
        click.echo(f"Description: {task['expected_output']}")
```

这让你可以先列出所有已记录的 Task ID，再选择从哪个开始 replay。

## 21.7 crewai deploy：云端部署

`deploy` 是一个命令组，包含多个子命令，用于将 Crew 部署到 CrewAI+ 平台：

```python
@crewai.group()
def deploy():
    """Deploy the Crew CLI group."""

@deploy.command(name="create")
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt")
def deploy_create(yes: bool):
    deploy_cmd = DeployCommand()
    deploy_cmd.create_crew(yes)

@deploy.command(name="push")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deploy_push(uuid: str | None):
    deploy_cmd = DeployCommand()
    deploy_cmd.deploy(uuid=uuid)
```

完整的部署子命令：

| 子命令 | 说明 |
|--------|------|
| `crewai deploy create` | 创建部署记录 |
| `crewai deploy push` | 推送代码并部署 |
| `crewai deploy list` | 列出所有部署 |
| `crewai deploy status` | 查看部署状态 |
| `crewai deploy logs` | 查看部署日志 |
| `crewai deploy remove` | 删除部署 |

`DeployCommand` 类在 `cli/deploy/main.py` 中，继承自 `BaseCommand` 和 `PlusAPIMixin`：

```python
class DeployCommand(BaseCommand, PlusAPIMixin):
    def __init__(self):
        BaseCommand.__init__(self)
        PlusAPIMixin.__init__(self, telemetry=self._telemetry)
        self.project_name = get_project_name(require=True)
```

部署流程：

1. **`deploy create`** — 读取 `.env` 文件和 Git 远程仓库 URL，通过 Plus API 创建部署记录
2. **`deploy push`** — 通过 UUID 或项目名触发部署，平台从 Git 仓库拉取代码并构建
3. **状态查询** — `deploy status` 和 `deploy logs` 通过 API 获取部署进度和日志

部署 payload 结构：

```python
payload = {
    "deploy": {
        "name": self.project_name,
        "repo_clone_url": remote_repo_url,
        "env": env_vars,
    }
}
```

## 21.8 crewai flow：Flow 专用命令

Flow 命令组包含三个子命令：

```python
@crewai.group()
def flow():
    """Flow related commands."""

@flow.command(name="kickoff")
def flow_run():
    """Kickoff the Flow."""
    kickoff_flow()

@flow.command(name="plot")
def flow_plot():
    """Plot the Flow."""
    plot_flow()

@flow.command(name="add-crew")
@click.argument("crew_name")
def flow_add_crew(crew_name):
    """Add a crew to an existing flow."""
    add_crew_to_flow(crew_name)
```

- **`flow kickoff`** — 执行 `uv run kickoff`，运行 Flow 的 `main()` 入口
- **`flow plot`** — 执行 `uv run plot`，生成 Flow 的可视化图形
- **`flow add-crew`** — 向现有 Flow 项目添加一个新的 Crew 子项目

## 21.9 crewai memory：Memory TUI

`memory` 命令启动一个基于 Textual 的终端 UI，用于浏览和查询 Memory 存储：

```python
@crewai.command()
@click.option("--storage-path", type=str, default=None,
    help="Path to LanceDB memory directory.")
@click.option("--embedder-provider", type=str, default=None)
@click.option("--embedder-model", type=str, default=None)
@click.option("--embedder-config", type=str, default=None,
    help='Full embedder config as JSON')
def memory(storage_path, embedder_provider, embedder_model, embedder_config):
    """Open the Memory TUI to browse scopes and recall memories."""
    app = MemoryTUI(storage_path=storage_path, embedder_config=embedder_spec)
    app.run()
```

TUI 支持：
- 浏览不同 scope 下的 Memory 记录
- 按关键词搜索 Memory
- 查看 Memory 的时间戳、分类等元数据

## 21.10 crewai reset-memories：重置记忆

```python
@crewai.command()
@click.option("-m", "--memory", is_flag=True, help="Reset MEMORY")
@click.option("-kn", "--knowledge", is_flag=True, help="Reset KNOWLEDGE storage")
@click.option("-akn", "--agent-knowledge", is_flag=True, help="Reset AGENT KNOWLEDGE")
@click.option("-k", "--kickoff-outputs", is_flag=True, help="Reset LATEST KICKOFF OUTPUTS")
@click.option("-a", "--all", is_flag=True, help="Reset ALL memories")
def reset_memories(memory, long, short, entities, knowledge, kickoff_outputs,
                   agent_knowledge, all):
    """Reset the crew memories."""
```

注意遗留标志的兼容处理：`--long`、`--short`、`--entities` 已被标记为 `hidden=True` 并打印废弃警告，统一映射到 `--memory`：

```python
if long or short or entities:
    click.echo(
        f"Warning: {', '.join(legacy_used)} deprecated. Use --memory (-m) instead."
    )
    memory = True
```

## 21.11 crewai chat：交互式对话

```python
@crewai.command()
def chat():
    """Start a conversation with the Crew."""
    click.secho("\nStarting a conversation with the Crew\n"
                "Type 'exit' or Ctrl+C to quit.\n")
    run_chat()
```

`crew_chat.py` 实现了一个完整的对话循环，使用 Chat LLM 根据 Crew 的配置信息生成回答。它会检查 CrewAI 版本是否支持对话模式（要求 >= 0.98.0）。

## 21.12 crewai uv：包管理器封装

```python
@crewai.command(name="uv",
    context_settings=dict(ignore_unknown_options=True))
@click.argument("uv_args", nargs=-1, type=click.UNPROCESSED)
def uv(uv_args):
    """A wrapper around uv commands that adds custom tool authentication."""
```

这个命令是对 `uv` 的透明封装，但在执行前会从 `pyproject.toml` 中读取 tool 仓库的认证信息并注入环境变量。`ignore_unknown_options=True` 和 `type=click.UNPROCESSED` 确保所有参数原封不动地传递给 `uv`。

## 21.13 crewai triggers：触发器管理

```python
@crewai.group()
def triggers():
    """Trigger related commands."""

@triggers.command(name="list")
def triggers_list():
    """List all available triggers from integrations."""

@triggers.command(name="run")
@click.argument("trigger_path")
def triggers_run(trigger_path: str):
    """Execute crew with trigger payload. Format: app_slug/trigger_slug"""
```

Trigger 支持通过外部集成（如 webhook、定时任务等）触发 Crew 执行。`trigger_path` 采用 `app_slug/trigger_slug` 格式标识具体的触发器。

## 21.14 crewai traces：追踪管理

追踪（tracing）命令组管理 Crew 执行的 trace 数据收集：

```python
@crewai.group()
def traces():
    """Trace collection management commands."""

@traces.command("enable")
def traces_enable():
    """Enable trace collection."""
    user_data = _load_user_data()
    user_data["trace_consent"] = True
    _save_user_data(user_data)

@traces.command("disable")
def traces_disable():
    """Disable trace collection."""

@traces.command("status")
def traces_status():
    """Show current trace collection status."""
```

追踪开关存储在用户数据文件中，同时也受环境变量 `CREWAI_TRACING_ENABLED` 控制。`traces status` 会综合显示环境变量、用户授权状态和最终的追踪启用状态。

## 21.15 其他辅助命令

### version

```python
@crewai.command()
@click.option("--tools", is_flag=True, help="Show the installed version of crewai tools")
def version(tools):
    crewai_version = get_version("crewai")
    click.echo(f"crewai version: {crewai_version}")
```

### install

```python
@crewai.command(context_settings=dict(
    ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
def install(context):
    """Install the Crew."""
    install_crew(context.args)
```

使用 `@click.pass_context` 获取额外参数并传递给安装函数。

### update

```python
@crewai.command()
def update():
    """Update the pyproject.toml of the Crew project to use uv."""
    update_crew()
```

用于将旧的 Poetry 格式项目迁移到 uv 格式。

### login

```python
@crewai.command()
def login():
    """Sign Up/Login to CrewAI AMP."""
    Settings().clear_user_settings()
    AuthenticationCommand().login()
```

## 21.16 CLI 命令执行模式总结

纵观整个 CLI 的实现，可以发现一个统一的模式：

```
CLI 命令 (Click)
  └─> 调用对应的 Python 函数
       └─> 通过 subprocess 执行 `uv run <script_name>`
            └─> 执行 pyproject.toml 中定义的脚本入口
                 └─> 调用 main.py 中的对应函数
                      └─> 创建 Crew 实例，调用 kickoff/train/replay/test
```

这种间接调用的设计保证了：

1. **环境隔离** — 通过 `uv run` 确保在正确的虚拟环境中执行
2. **项目解耦** — CLI 不直接依赖用户项目代码，通过 subprocess 调用
3. **灵活性** — 用户可以自定义 `main.py` 中的入口函数，CLI 只负责触发

## 本章要点

- CrewAI CLI 基于 Click 框架构建，采用分层命令组设计，顶层入口是 `crewai()` 组
- `crewai create crew/flow` 通过模板系统生成标准项目结构，包含 `pyproject.toml`、YAML 配置、Python 源码等
- 项目脚手架自动处理名称转换（folder_name、class_name），并验证 Python 标识符合法性
- `crewai run` 自动检测项目类型（Crew / Flow），通过 `uv run` 执行对应脚本
- `crewai train` 支持多轮迭代训练，训练数据保存为 `.pkl` 文件
- `crewai test` 通过指定 LLM 对 Crew 输出进行自动评估，experimental 模块提供更完整的实验框架
- `crewai replay` 利用 SQLite 存储的 Task 输出，支持从任意 Task 开始重新执行
- `crewai deploy` 命令组通过 CrewAI+ API 管理云端部署的完整生命周期
- `crewai memory` 提供基于 Textual 的 TUI 浏览器，用于查看和搜索 Memory 存储
- CLI 的统一执行模式是：Click 命令 -> subprocess `uv run` -> pyproject.toml 脚本入口 -> 用户代码
