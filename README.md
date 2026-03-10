# CrewAI 源码解析

> 深入剖析 CrewAI —— 多 Agent 协作编排框架

## 关于本书

CrewAI 是一个基于角色驱动的多 Agent 协作框架。本书从源码层面深度解析 CrewAI 的架构设计与实现细节，覆盖核心三元组（Agent/Task/Crew）、Flow 事件驱动工作流、LLM 抽象层、记忆与知识系统、工具生态、A2A 通信协议等完整知识体系。

## 适合读者

- Python 开发者，希望深入理解多 Agent 编排框架
- AI 工程师，需要在生产环境中部署 CrewAI
- CrewAI 贡献者，需要理解框架内部实现
- 对比学习者，希望了解 CrewAI 与 LangGraph 等框架的设计差异

## 内容结构

全书 **22 章 + 3 附录**，分 8 个部分：

1. **宏观认知**（2 章）：项目概览、仓库结构
2. **核心三元组**（4 章）：Agent、Task、Crew、Output
3. **Flow 工作流**（4 章）：装饰器模型、执行引擎、持久化、可视化
4. **LLM 抽象层**（3 章）：统一接口、Hooks、LiteLLM 集成
5. **记忆与知识**（2 章）：Memory 系统、Knowledge RAG
6. **工具生态**（2 章）：Tools 框架、MCP 集成
7. **协议与安全**（3 章）：A2A、Events、Security
8. **CLI 与生产化**（2 章）：CLI 工具链、Project 模板

## 阅读建议

| 阶段 | 章节 | 说明 |
|------|------|------|
| 快速了解 | 第 1-2 章 | 理解 CrewAI 定位与全局结构 |
| 核心概念 | 第 3-6 章 | 掌握 Agent/Task/Crew 三元组 |
| 进阶工作流 | 第 7-10 章 | 理解 Flow 事件驱动编排 |
| 深入内部 | 第 11-17 章 | LLM、Memory、Knowledge、Tools |
| 生产部署 | 第 18-22 章 | 协议、安全、CLI、项目模板 |

## 源码版本

基于 CrewAI v1.10.1 源码分析。

## 在线阅读

[https://coolclaws.github.io/crewai-book/](https://coolclaws.github.io/crewai-book/)

## 许可证

- 书籍内容：CC BY-NC-SA 4.0
- CrewAI 项目：MIT License
