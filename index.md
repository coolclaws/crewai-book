---
layout: home

hero:
  name: "CrewAI 源码解析"
  text: "深入剖析多 Agent 协作编排框架"
  tagline: 从 Agent 角色设定到 Flow 事件驱动工作流，全面解读 CrewAI 的架构设计与实现细节
  actions:
    - theme: brand
      text: 开始阅读
      link: /chapters/01-overview
    - theme: alt
      text: 查看目录
      link: /contents
    - theme: alt
      text: GitHub
      link: https://github.com/coolclaws/crewai-book

features:
  - icon:
      src: /icons/architecture.svg
    title: 架构全景
    details: 从 Crew 团队编排到 Task 任务执行，系统梳理 CrewAI 的核心三元组设计，理解角色驱动的多 Agent 协作模式。

  - icon:
      src: /icons/flow.svg
    title: Flow 工作流深挖
    details: 深入 Flow 事件驱动引擎，剖析 @start/@listen/@router 装饰器模型、状态持久化与可视化的完整实现。

  - icon:
      src: /icons/tools.svg
    title: 工具与协议生态
    details: 覆盖 BaseTool 框架、MCP 集成、A2A 通信协议，理解 CrewAI 的开放式工具生态与跨 Agent 通信设计。

  - icon:
      src: /icons/security.svg
    title: 记忆、安全与生产化
    details: Memory 四种记忆类型、Knowledge RAG 系统、Security 指纹机制、CLI 工具链——补全生产级部署所需的完整知识体系。
---
