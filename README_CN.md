<div align="center">

# Model Price

**AI 模型定价聚合器** - 一站式比较各大云服务商的 AI 模型价格

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb?style=flat-square&logo=react&logoColor=white)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178c6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[English](README.md) | [简体中文](README_CN.md)

</div>

---

## 功能亮点

- **多平台聚合** - 整合 6 大主流 AI 服务商的模型定价
- **实时更新** - 自动爬取官方定价数据，保持信息新鲜
- **智能筛选** - 按提供商、模型系列、能力标签快速定位
- **双视图模式** - 卡片视图与表格视图自由切换
- **价格对比** - 直观的价格柱状图，一眼看清性价比
- **574+ 模型** - 覆盖 GPT、Claude、Gemini、Llama 等主流模型

## 界面预览

<div align="center">

#### 卡片视图
![卡片视图](docs/images/main-view.png)

#### 表格视图
![表格视图](docs/images/table-view.png)

#### 筛选功能
![筛选功能](docs/images/filter-provider.png)

</div>

## 支持的服务商

| 服务商 | 模型数量 | 数据来源 | 更新方式 |
|:------|:-------:|:--------|:--------|
| **AWS Bedrock** | 96+ | 公开定价 API | 自动 |
| **Azure OpenAI** | 50+ | 零售价格 API | 自动 |
| **OpenAI** | 53+ | 官网爬虫 | 自动 |
| **Google Gemini** | 31+ | 官网爬虫 | 自动 |
| **OpenRouter** | 339+ | 公开 API | 自动 |
| **xAI (Grok)** | 12+ | 官方文档 | 手动 |

## 技术栈

<table>
<tr>
<td align="center" width="50%">

**后端**

</td>
<td align="center" width="50%">

**前端**

</td>
</tr>
<tr>
<td>

- Python 3.11+
- FastAPI
- Playwright (网页爬虫)
- httpx (异步 HTTP)
- uv (包管理器)

</td>
<td>

- React 18
- TypeScript 5
- Vite
- CSS Variables (主题系统)

</td>
</tr>
</table>

## 快速开始

### 方式一：本地开发

```bash
# 1. 克隆仓库
git clone https://github.com/xiaobox/model-price.git
cd model-price

# 2. 启动后端
cd backend
uv run main.py
# API 服务运行在 http://localhost:8000

# 3. 启动前端 (新终端)
cd frontend
npm install
npm run dev
# 前端运行在 http://localhost:5173
```

### 方式二：Docker 部署

```bash
# 即将支持
docker-compose up -d
```

## API 文档

启动后端后访问 http://localhost:8000/docs 查看完整的 Swagger API 文档。

### 核心接口

| 方法 | 路径 | 描述 |
|:-----|:-----|:-----|
| `GET` | `/api/models` | 获取所有模型（支持筛选和排序） |
| `GET` | `/api/models/{id}` | 获取单个模型详情 |
| `GET` | `/api/providers` | 获取提供商列表 |
| `GET` | `/api/families` | 获取模型系列列表 |
| `GET` | `/api/stats` | 获取统计信息 |
| `POST` | `/api/refresh` | 刷新定价数据 |

## 项目结构

```
model-price/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置管理
│   ├── models/              # 数据模型
│   ├── providers/           # 各服务商数据获取器
│   │   ├── aws_bedrock.py
│   │   ├── azure_openai.py
│   │   ├── openai.py
│   │   ├── google_gemini.py
│   │   ├── openrouter.py
│   │   └── xai.py
│   ├── services/            # 业务逻辑
│   └── data/                # 缓存数据
├── frontend/
│   ├── src/
│   │   ├── components/      # React 组件
│   │   ├── hooks/           # 自定义 Hooks
│   │   ├── config/          # 前端配置
│   │   └── types/           # TypeScript 类型
│   └── package.json
└── docs/
    └── images/              # 文档图片
```

## 数据更新策略

| 服务商 | 技术方案 | 认证需求 | 可靠性 |
|:------|:--------|:--------|:------:|
| AWS Bedrock | httpx 异步请求 | 无需 | 高 |
| Azure OpenAI | httpx + 分页 | 无需 | 高 |
| OpenAI | Playwright 爬虫 | 无需 | 中 |
| Google Gemini | Playwright 爬虫 | 无需 | 中 |
| OpenRouter | httpx API | 无需 | 高 |
| xAI | 静态数据 | N/A | 需手动更新 |

## 开发指南

### 后端开发

```bash
cd backend

# 添加依赖
uv add <package-name>

# 运行开发服务器（热重载）
uv run main.py

# 手动刷新数据
curl -X POST http://localhost:8000/api/refresh
```

### 前端开发

```bash
cd frontend

# 安装依赖
npm install

# 开发模式
npm run dev

# 生产构建
npm run build
```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 [MIT License](LICENSE) 开源。

## 支持我们

如果本项目对你有所帮助，可以通过以下方式支持我们的持续开发。

<table style="margin: 0 auto">
  <tbody>
    <tr>
      <td align="center" style="width: 260px">
        <img
          src="https://xiaobox-public-images.oss-cn-beijing.aliyuncs.com/imagescc16a59f8b43da4a3ad3ce201f46fc9d.jpg"
          style="width: 200px"
        /><br />
      </td>
      <td align="center" style="width: 260px">
        <img
          src="https://xiaobox-public-images.oss-cn-beijing.aliyuncs.com/images2d585d78e23826f6698ddd4edec5d9c2.jpg"
          style="width: 200px"
        /><br />
      </td>
    </tr>
  </tbody>
</table>

---

如果这个项目对你有帮助，欢迎 Star ⭐️ 支持！也欢迎提交 Issue/PR 一起把它打磨得更好。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=xiaobox/model-price&type=Date)](https://www.star-history.com/#xiaobox/model-price&Date)
