import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'CrewAI 源码解析',
  description: '深入剖析 CrewAI —— 多 Agent 协作编排框架',
  lang: 'zh-CN',

  base: '/',

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/crewai-book/favicon.svg' }],
    ['link', { rel: 'icon', type: 'image/x-icon', href: '/crewai-book/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#f97316' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'CrewAI 源码解析' }],
    ['meta', { property: 'og:description', content: '深入剖析 CrewAI —— 多 Agent 协作编排框架' }],
  ],

  themeConfig: {
    logo: { src: '/logo.png', alt: 'CrewAI' },

    nav: [
      { text: '开始阅读', link: '/chapters/01-overview' },
      { text: '目录', link: '/contents' },
      { text: 'GitHub', link: 'https://github.com/coolclaws/crewai-book' },
    ],

    sidebar: [
      {
        text: '前言',
        items: [
          { text: '关于本书', link: '/' },
          { text: '完整目录', link: '/contents' },
        ],
      },
      {
        text: '第一部分：宏观认知',
        collapsed: false,
        items: [
          { text: '第 1 章　项目概览与设计哲学', link: '/chapters/01-overview' },
          { text: '第 2 章　仓库结构与模块地图', link: '/chapters/02-repo-structure' },
        ],
      },
      {
        text: '第二部分：核心三元组',
        collapsed: false,
        items: [
          { text: '第 3 章　Agent：角色设定与执行逻辑', link: '/chapters/03-agent' },
          { text: '第 4 章　Task：任务定义与 Guardrail', link: '/chapters/04-task' },
          { text: '第 5 章　Crew：团队编排与 Process', link: '/chapters/05-crew' },
          { text: '第 6 章　输出系统：CrewOutput 与结构化输出', link: '/chapters/06-output' },
        ],
      },
      {
        text: '第三部分：Flow 事件驱动工作流',
        collapsed: false,
        items: [
          { text: '第 7 章　Flow 基础：装饰器模型', link: '/chapters/07-flow-basics' },
          { text: '第 8 章　Flow 执行引擎', link: '/chapters/08-flow-engine' },
          { text: '第 9 章　Flow 持久化', link: '/chapters/09-flow-persistence' },
          { text: '第 10 章　Flow 可视化', link: '/chapters/10-flow-visualization' },
        ],
      },
      {
        text: '第四部分：LLM 抽象层',
        collapsed: false,
        items: [
          { text: '第 11 章　统一 LLM 接口', link: '/chapters/11-llm-interface' },
          { text: '第 12 章　LLM Hooks 与 Tool Hooks', link: '/chapters/12-llm-hooks' },
          { text: '第 13 章　LiteLLM 集成与模型路由', link: '/chapters/13-litellm' },
        ],
      },
      {
        text: '第五部分：记忆与知识',
        collapsed: false,
        items: [
          { text: '第 14 章　Memory 系统', link: '/chapters/14-memory' },
          { text: '第 15 章　Knowledge：文档摄入与 RAG', link: '/chapters/15-knowledge' },
        ],
      },
      {
        text: '第六部分：工具生态',
        collapsed: false,
        items: [
          { text: '第 16 章　Tools 框架', link: '/chapters/16-tools' },
          { text: '第 17 章　MCP 集成', link: '/chapters/17-mcp' },
        ],
      },
      {
        text: '第七部分：协议、可观测性与安全',
        collapsed: false,
        items: [
          { text: '第 18 章　A2A 通信协议', link: '/chapters/18-a2a' },
          { text: '第 19 章　Events 系统', link: '/chapters/19-events' },
          { text: '第 20 章　Security 与 Telemetry', link: '/chapters/20-security' },
        ],
      },
      {
        text: '第八部分：CLI 与生产化',
        collapsed: false,
        items: [
          { text: '第 21 章　CLI：创建、训练与部署', link: '/chapters/21-cli' },
          { text: '第 22 章　Project 模板与 @CrewBase', link: '/chapters/22-project' },
        ],
      },
      {
        text: '附录',
        collapsed: true,
        items: [
          { text: '附录 A：推荐阅读路径', link: '/chapters/appendix-a-reading-path' },
          { text: '附录 B：关键类型速查', link: '/chapters/appendix-b-type-reference' },
          { text: '附录 C：名词解释（Glossary）', link: '/chapters/appendix-c-glossary' },
        ],
      },
    ],

    outline: {
      level: [2, 3],
      label: 'On This Page',
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/coolclaws/crewai-book' },
    ],

    footer: {
      message: '基于 MIT 协议发布',
      copyright: 'Copyright © 2025-present',
    },

    search: {
      provider: 'local',
    },
  },

  markdown: {
    lineNumbers: true,
  },
})
