# Windows GUI 迁移分析

> 阶段：Phase 1 仓库分析
> 分析日期：2026-08-14
> 当前主线：CTM2.0 / `ac_icam_real_outcome_pfs_v8`
> 本阶段约束：只分析和规划，不重构界面、不修改模型核心、不建立桌面宿主。

## 1. 执行结论

本项目适合采用“现有 WebUI + WebView2 + WinUI 3 宿主 + Python 本地引擎”的渐进迁移方案，但需求文档中的一个前提需要纠正：**当前正式 WebUI 不是 React 应用**。当前实现是：

- Flask/Jinja 生成单页 HTML；
- `static/app.css` 负责完整视觉样式；
- 原生 JavaScript 负责主要交互和结果渲染；
- TypeScript 已用于口腔腺瘤模块、API 类型和构建入口，但主界面逻辑仍在 `frontend/src/legacy-app.js`；
- esbuild 将 TypeScript 与现有 JavaScript 打包为 `static/generated/app.js`；
- 没有 React、前端路由库或集中式状态管理库。

因此，不应为了符合架构文档中的“React”字样而重写前端。那样会违反“保留现有 WebUI、最小视觉改动”的最高优先级约束。桌面迁移应直接复用现有 HTML、CSS、DOM 结构和渲染逻辑，在外围增加 TypeScript 传输适配层和 C# 宿主。

综合当前代码，预计：

| 范围 | 可保留比例 | 说明 |
|---|---:|---|
| 用户可见布局、配色、卡片、表单和结果样式 | **95% 以上** | `templates/index.html` 与 `static/app.css` 可基本原样使用 |
| 当前前端交互和结果渲染源码 | **约 85%–90%** | API 调用、文件保存、外链和错误传输需要适配 |
| Python 预处理、模型和药学决策逻辑 | **接近 100%** | 应在外层增加 Service/API，不重写算法 |
| 当前 Windows 系统能力 | **0%** | 仓库中尚无 C#、WinUI 3、WebView2、SQLite 或安装工程 |

这些比例是迁移工作量估算，不是测试覆盖率或产品完成度。

## 2. 分析范围与方法

本次检查覆盖了当前活动源码、配置、文档、测试、前端构建、模型入口和持久化路径。`node_modules/`、`outputs/` 中的大量生成文件和 `archive/` 中的历史实现按生成物或归档边界分类，没有逐文件作为现行产品代码阅读。

本阶段没有执行以下操作：

- 没有修改 WebUI；
- 没有修改 Python 模型或模型工件；
- 没有移动现有目录；
- 没有创建 C# 项目；
- 没有引入 FastAPI、SQLite 或新依赖；
- 没有删除、覆盖或归档任何本地文件。

## 3. 当前技术架构

### 3.1 现行运行链

```text
用户浏览器
  -> Flask GET /
     -> Jinja 渲染 templates/index.html
     -> 加载 static/app.css
     -> 加载 static/generated/app.js
  -> POST /standardize 或 POST /analyze
     -> 临床字段标准化与校验
     -> src.pipeline.run_pipeline()
        -> 结构化预处理
        -> 微生物图构建与描述性拓扑特征
        -> 当前模型桥接
        -> 药学辅助决策
        -> 结构化报告
     -> Python 将报告 JSON 写入 outputs/
     -> 返回结果与保存路径
```

### 3.2 已验证技术栈

| 层 | 当前技术 | 关键文件 |
|---|---|---|
| 页面 | Jinja HTML 单页 | `templates/index.html` |
| 样式 | 原生 CSS | `static/app.css` |
| 前端交互 | 原生 JavaScript + 渐进式 TypeScript | `frontend/src/legacy-app.js`, `frontend/src/*.ts` |
| 前端构建 | TypeScript 5.9 + esbuild | `package.json`, `frontend/tsconfig.json` |
| HTTP 后端 | Flask 3.0.3 | `enhanced_app.py` |
| 输入处理 | Python、Pandas、NumPy | `src/clinical_standardizer.py`, `src/preprocess.py`, `src/validators.py` |
| 图处理 | NetworkX；旧研究路径含 PyTorch Geometric | `src/graph_builder.py`, `src/temporal_topology_bridge.py` |
| 当前 PFS 模型 | AC-ICAM V8 五成员 ridge-Cox | `src/ac_icam_v8_bridge.py` |
| 无病理研究指数 | 时间拓扑/GNN/AFT 旧研究后端 | `src/temporal_topology_bridge.py` |
| 口腔腺瘤模型 | 独立内部研究端点，默认关闭 | `src/oral_adenoma_bridge.py` |
| 药学辅助 | 版本化规则和本地药品证据 | `src/pharmacy_engine.py`, `src/drug_knowledge.py` |
| 当前持久化 | 文件系统 JSON | `src/export_utils.py`, `outputs/` |
| 当前日志 | Python 标准错误流 | `src/logging_utils.py` |
| 桌面技术 | 尚未实现 | 当前没有 `.cs`, `.csproj`, `.sln` 或 `.xaml` 文件 |

本机已检测到 .NET SDK 10.0.302 和 WebView2 Runtime，但仓库尚无 WinUI 3 工程。正式实现时仍需建立并锁定经验证的 Windows App SDK、WebView2 SDK 和目标 .NET 版本，不能依赖开发机的隐式全局状态。

### 3.3 仓库边界

| 目录/文件 | 当前角色 | 桌面发布建议 |
|---|---|---|
| `templates/`, `static/`, `frontend/` | 正式 WebUI | 保留并构建为桌面静态资源 |
| `enhanced_app.py` | 当前 Flask 页面和 API 入口 | 保留为浏览器兼容入口，逐步抽离 API 职责 |
| `src/` | 当前运行时预处理、模型桥接、报告和药学逻辑 | 保留；由新 AI Service 调用 |
| `config/releases/` | 本地发布工件和版本锁 | 建立发布清单并随安装包受控交付 |
| `data/pharmacy_knowledge/` | 药学运行时知识库 | 只打包运行所需、许可明确、哈希锁定的数据 |
| `research/`, `experiments/` | 训练和科研评估 | 默认不进入终端用户安装包 |
| `outputs/` | 本地实验结果和当前报告输出 | 不整体打包；只选择运行时必需工件 |
| `ctm_fusion_experiment/` | 历史独立实验 | 不进入正式桌面运行时 |
| `archive/` | 旧版本与回滚证据 | 保留在仓库，不进入默认安装包 |

## 4. 当前 WebUI 清单

### 4.1 页面和路由

当前只有一个正式页面，由 `GET /` 返回。没有前端路由，也没有多页面导航、侧边栏或顶部菜单。

现有单页包含以下功能区：

1. 产品标题、当前模型和研究用途边界；
2. JSON 文件导入、示例载入、标准化预览；
3. 五项核心菌群和自定义菌种输入；
4. 年龄、性别及可选结直肠癌病理信息；
5. BMI、吸烟、家族史和代谢特征；
6. 当前用药、过敏、抗生素/益生菌、肝肾功能和妊娠背景；
7. 可选的 381 菌属口腔腺瘤内部研究面板；
8. 风险数值、风险刻度、36/60 月 PFS 和模型适用范围；
9. 药学状态、优先行动、后续核对和药品标签证据；
10. 模型关注微生物、报告路径及完整研究 JSON。

需求文档中提到的下列页面目前并不存在，因此它们属于未来新增业务，不属于“原样迁移”范围：

- 独立首页或工作台；
- 患者管理；
- 检测记录管理；
- 报告中心；
- 系统设置；
- 用户和权限管理；
- 设备管理。

这些页面以后可以复用当前卡片、表单、按钮和配色体系，但不应在桌面宿主初版中假装已经存在。

### 4.2 主要前端组件和状态方式

| 功能 | 当前实现 | 状态来源 |
|---|---|---|
| JSON 导入 | 文件输入、文本框、标准化按钮 | DOM 元素值 |
| 临床/菌群表单 | 固定字段 + 动态菌种行 | DOM 元素值 |
| 分析请求 | 相对路径 `fetch()` | Promise 与状态提示框 |
| 风险展示 | 横幅、数值、刻度和摘要卡片 | API 返回 JSON |
| 药学建议 | 动态创建卡片、列表、证据链接 | API 返回 JSON |
| 口腔腺瘤 | TypeScript 模块 | 模块缓存 + DOM |
| 文件下载 | `Blob` + Object URL + 隐式 `<a>` 点击 | 浏览器下载能力 |
| 页面启动 | `DOMContentLoaded` | 无集中式 store |

当前没有 Redux、Zustand、Pinia、React Context 或客户端路由。桌面迁移不需要为了“工程化”强制引入这些库。

### 4.3 前端源码现状

- `frontend/src/main.ts` 直接导入整个 `legacy-app.js`，再初始化口腔腺瘤 TypeScript 模块；
- `frontend/src/legacy-app.js` 与 `static/app.js` 当前内容和 SHA-256 完全一致；
- 页面实际加载的是构建产物 `static/generated/app.js`；
- `packages/client/` 是独立 TypeScript API 客户端包，当前浏览器页面没有直接使用它；
- 主分析响应类型仍大量使用 `Record<string, unknown>`，后续可以渐进收紧，但不能以此为由重写 UI。

后续应明确单一前端源码入口，避免同时手工维护 `static/app.js` 和 `frontend/src/legacy-app.js`。这是维护性改进，不是视觉重构。

## 5. A–E 迁移分类

分类优先级遵循 `A > B > C > D > E`。

### A. 直接保留

以下内容可以直接进入 WebView2，原则上不改视觉和布局：

- `templates/index.html` 的页面层级和全部现有功能区；
- `static/app.css` 的配色、字体回退、栅格、卡片、按钮、状态和响应式样式；
- 风险横幅、风险刻度、PFS 摘要、药学建议卡片和折叠技术详情；
- 临床、菌群、代谢和药学背景表单的布局；
- 口腔腺瘤面板的现有视觉；
- 当前加载、成功、警告和错误状态的视觉表达；
- 主要结果渲染函数和输入解析函数；
- 当前中文医学边界提示。

### B. 保留界面，仅修改数据来源

- `/standardize` 和 `/analyze` 的调用改为版本化传输适配器；
- 口腔腺瘤 schema 和 analyze 请求改为统一桥接；
- “报告保存位置”由 Python 相对路径改为 C# 管理的报告记录或用户路径；
- 当前模型、模型加载状态和版本信息改为宿主启动配置；
- 技术错误改为稳定错误码，页面仍使用现有状态框展示通俗提示；
- 未来患者、样本和报告列表由 C# SQLite 服务供数，但视觉应沿用现有体系。

### C. 保留界面，底层功能交给 C#

- Windows 应用启动、退出和单实例生命周期；
- Python Engine 启动、健康检查、超时、崩溃恢复和退出；
- WebView2 初始化、导航限制、权限限制和外链处理；
- 打开 JSON 文件、保存模板、选择报告目录；
- 报告索引、PDF 导出、打印和文件路径校验；
- SQLite、事务、迁移和备份；
- 本地配置、应用/前端/引擎/模型/数据库版本；
- 文件日志、崩溃日志和敏感信息脱敏；
- 软件更新；
- 未来 USB、串口和检测设备通信。

### D. 保留界面，算法继续收敛在 Python

- 临床原始字段标准化和输入校验；
- 结构化预处理；
- 微生物图构建与拓扑特征；
- AC-ICAM V8 PFS 推理；
- 无病理时间拓扑研究指数；
- 口腔腺瘤内部研究模型；
- Cox、GNN、AFT 及模型工件解析；
- 风险分层、PFS 概率和可靠性信息；
- 药物标准化、有限相互作用筛查、标签证据和药学建议；
- 结构化报告内容生成。

### E. 需要小范围重构

只有以下技术问题需要定向重构：

1. 将前端直接 `fetch()` 封装为 Web/桌面双传输接口；
2. 将“推理”和“写入报告文件”拆开；
3. 增加版本化 FastAPI Service 层，同时保留现有 Flask 兼容接口；
4. 将 Jinja 的少量启动变量改为可由构建或 C# 注入的 bootstrap 配置；
5. 统一结构化错误和请求追踪 ID；
6. 明确前端单一源码，消除重复 JavaScript 的双写风险；
7. 将生产推理所需的旧研究模块和工件整理为明确清单，避免打包整个 `research/` 和 `outputs/`。

上述重构都不要求改变页面视觉、信息层级或模型数学逻辑。

## 6. 当前 API、模型和持久化

### 6.1 当前 HTTP 接口

| 方法 | 路径 | 当前职责 | 问题 |
|---|---|---|---|
| GET | `/` | Jinja 页面 | 不能直接作为最终静态桌面资源 |
| POST | `/standardize` | 标准化和校验 | 未版本化，错误契约为 `ok/errors` |
| POST | `/analyze` | 推理、药学、报告组装并写文件 | 计算与持久化耦合 |
| GET | `/internal/oral-adenoma/schema` | 可选模型元数据 | 路径和错误未版本化 |
| POST | `/internal/oral-adenoma/analyze` | 可选内部研究推理 | 路径和错误未版本化 |

当前没有 `/health`，也没有 FastAPI。`enhanced_app.py` 会依次尝试多个固定端口；桌面宿主不能沿用这种不确定端口策略。

### 6.2 实际模型路由

需求文档中的“每次输入都经过 GNN -> COX”与当前正式模型不完全一致。实际情况是：

```text
完整肿瘤病理资料
  -> AC-ICAM V8 clinical core ridge-Cox
  -> 若有肿瘤 RNA 实测 ICR，则使用 ICR 扩展 ridge-Cox

病理资料不完整 + 五项核心菌群完整
  -> 时间拓扑/GNN/AFT 研究参考指数
  -> 不输出一般人群患癌概率，也不计算 PFS

所有适用输入
  -> 独立药学辅助层
```

桌面端必须保持这一路由和医学边界，不能为了统一架构图而让所有请求强行经过 GNN。

### 6.3 Python 核心模块

| 模块 | 作用 | 迁移原则 |
|---|---|---|
| `src/pipeline.py` | 当前统一分析入口 | 作为 AI Service 的内部调用入口 |
| `src/validators.py` | 输入范围和完整性校验 | 保留；API schema 再做外层校验 |
| `src/clinical_standardizer.py` | 原始临床 JSON 标准化 | 保留 |
| `src/preprocess.py` | 结构化特征预处理 | 不改算法 |
| `src/graph_builder.py` | 图构建和描述性拓扑 | 不改算法 |
| `src/ac_icam_v8_bridge.py` | 正式 V8 PFS 推理 | 重点保护，继续哈希校验 |
| `src/temporal_topology_bridge.py` | 无病理研究指数和旧后端 | 保留功能，单独解决打包依赖 |
| `src/oral_adenoma_bridge.py` | 可选内部口腔模型 | 保持独立，不与 PFS 合并 |
| `src/pharmacy_engine.py` | 药学辅助决策 | 保留规则边界和证据版本 |
| `src/drug_knowledge.py` | 药品知识和有限 DDI 数据 | 保留并建立发布清单 |
| `src/report.py` | 结构化报告内容 | 保留；文件保存交给 C# |

### 6.4 需要优先处理的隐性依赖

当前默认后端虽然是轻量的 V8，但在病理资料不完整且五菌完整时，`src/ac_icam_v8_bridge.py` 会动态调用 `src/temporal_topology_bridge.py`。该路径引入：

- PyTorch；
- PyTorch Geometric；
- XGBoost；
- Scikit-learn；
- 多个 `research/` 和 `experiments/` 模块；
- `outputs/current_mainline_v2/` 下的本地模型工件。

本机 `outputs/current_mainline_v2/` 当前约 1.14 GB，不能直接整体复制进安装包。必须生成“运行时工件清单”，只纳入无病理研究指数真正使用的检查点、校准器、参考分布和配置，并为每项记录 SHA-256、模型版本和用途。

这不是模型问题，也不应通过删除该功能来掩盖。它是桌面打包和运行时边界问题。

### 6.5 当前持久化方式

当前没有 SQLite、SQLAlchemy、数据库迁移或患者数据表。现有持久化只有：

- `/analyze` 每次调用通过 `src/export_utils.py` 写入时间戳 JSON；
- `clinical_workflow.py` 写标准化输入和报告 JSON；
- 模型、配置和药学知识从 JSON/YAML/模型文件读取；
- 训练和研究流程向 `outputs/` 写入大量 CSV、JSON、NPZ、PT 等工件；
- 日志仅输出到进程流，没有统一文件、保留策略或患者信息脱敏器。

因此 SQLite 是新增基础设施，不是现有数据库迁移。

## 7. 当前耦合点

| 耦合点 | 现状 | 迁移处理 |
|---|---|---|
| UI -> Flask | 前端写死相对 API 路径 | 增加 `Transport` 接口，不改渲染逻辑 |
| Flask -> Pipeline | 路由直接调用 `run_pipeline()` | 新增 AI Service，Flask/FastAPI 共用 |
| 推理 -> 文件系统 | `/analyze` 总是写 `outputs/` | 改为纯结构化响应；C# 决定是否保存 |
| 模型配置 -> 环境变量 | 模块导入时读取后端设置 | 由 Engine 启动配置固定并报告版本 |
| V8 -> 时间拓扑 | 无病理研究指数动态加载重模型 | 建立显式 capability 和工件清单 |
| UI -> 浏览器下载 | Blob 和隐藏链接下载模板 | 桌面模式交给 C# SaveFilePicker |
| UI -> 外部网页 | 药品证据使用 `_blank` | C# 白名单后交系统浏览器打开 |
| HTML -> Flask/Jinja | 标题、模型版本和功能开关由模板注入 | 改为 bootstrap 配置，保持同一 HTML 视觉 |
| Python -> 当前工作目录 | `outputs/` 等相对路径 | 桌面运行统一使用明确 AppData 路径 |

当前前端没有直接导入 Python 函数，也没有直接操作数据库，这一点符合目标职责边界，应继续保持。

## 8. WebView2 兼容性审查

| 功能 | WebView2 现状 | 结论 |
|---|---|---|
| HTML/CSS/DOM | 标准能力 | 可直接保留 |
| 响应式布局 | 标准 CSS | 可保留，需补窗口尺寸回归测试 |
| 相对路径 `fetch()` | 加载同源 Flask 页面时可用 | 桌面静态资源模式需传输适配 |
| JSON 文件输入和 `file.text()` | WebView2 支持 | 原型可用，正式版建议 C# 控制文件选择 |
| Blob 模板下载 | WebView2 可触发但路径不可控 | 改为 C# 保存对话框 |
| `_blank` 证据链接 | 可能创建未受控新窗口 | 拦截 `NewWindowRequested` 并使用白名单 |
| Jinja 变量 | 静态 WebView2 不解析 | 构建时或宿主启动时注入 bootstrap |
| 本地存储 | 当前未使用 | 无迁移负担 |
| Service Worker/WebSocket | 当前未使用 | 无迁移负担 |
| 浏览器打印 | 当前未使用 | 后续由 C# 报告/打印服务承担 |
| 外部字体/资源 | 当前没有网络字体依赖 | 有利于离线部署 |

正式 WebView2 必须：

- 只加载应用自有静态域或受控本机页面；
- 拦截并拒绝任意导航；
- 禁止任意脚本调用系统资源；
- 发布版关闭开发者工具、默认右键菜单和不需要的权限；
- 只允许 schema 明确、方法白名单明确的消息；
- 对每个请求设置 ID、超时和最大载荷；
- 不向页面暴露任意文件路径、任意命令或任意 URL 请求能力。

## 9. 推荐目标架构

```text
frontend/dist（现有 HTML/CSS/JS 的桌面构建产物）
  -> WebView2
     -> 受控 WebView2 Message Transport
        -> C# Message Router（方法白名单 + schema 校验）
           -> 文件、SQLite、报告、打印、设备和应用服务
           -> C# AI Engine Client
              -> 127.0.0.1 + 每次启动随机令牌
                 -> FastAPI AI Engine
                    -> 新增 AI Service 外层
                       -> 现有 src.pipeline / 模型 / 药学模块
```

浏览器开发模式继续使用 HTTP Transport；桌面发布模式使用 WebView2 Message Transport。两种传输返回同一 TypeScript 领域对象，因此现有页面渲染代码不需要感知 C#、端口或 Python 进程。

### 9.1 为什么最终建议由 C# 转发 AI 请求

WebView2 页面直接访问随机本机端口虽然实现简单，但需要处理 CORS，并会把本地 API 令牌暴露给页面。推荐的最终结构是：

1. WebUI 只向 C# 发送允许的业务命令；
2. C# 校验命令和请求大小；
3. C# 使用 `HttpClient` 调用仅监听 `127.0.0.1` 的 Python API；
4. C# 将结构化结果回传 WebUI；
5. Python 端口和启动令牌不暴露给普通页面逻辑。

开发阶段仍可保留浏览器直接访问 Flask/FastAPI 的方式，便于独立调试。

## 10. Python AI Engine 标准化方案

### 10.1 新 Service 层

新增薄层，不移动或改写现有模型：

```text
FastAPI route
  -> request schema
  -> AIService.standardize()/predict()/analyze()
  -> src.pipeline.run_pipeline()
  -> response schema
```

`AIService` 负责模型预加载、能力状态、调用计时和异常翻译；模型公式、权重和特征处理继续由现有 `src/` 模块负责。

### 10.2 推荐版本化接口

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/v1/health` | 进程、模型、能力和版本状态 |
| POST | `/api/v1/standardize` | 输入标准化和字段错误 |
| POST | `/api/v1/predict` | 纯模型预测，不写文件或数据库 |
| POST | `/api/v1/analyze` | 风险、药学和结构化报告内容，不写文件 |
| GET | `/api/v1/oral-adenoma/schema` | 可选内部模型 schema |
| POST | `/api/v1/oral-adenoma/analyze` | 可选内部模型推理 |

若保留 `/api/v1/report`，它只应生成结构化报告内容；PDF、打印、索引和最终文件保存仍由 C# 负责。

现有 `/standardize`、`/analyze` 和内部端点应暂时作为兼容适配器保留，直到浏览器版和桌面版均完成切换。不要一次性破坏现有 API。

### 10.3 健康检查

建议返回：

```json
{
  "status": "ok",
  "engine_ready": true,
  "model_loaded": true,
  "ai_engine_version": "0.1.0",
  "model_versions": {
    "pfs": "ac_icam_real_outcome_pfs_v8",
    "pharmacy": "pharmacy_assistance_v3"
  },
  "capabilities": {
    "pfs": true,
    "general_risk": true,
    "oral_adenoma": false
  }
}
```

`model_loaded` 不能只是进程在线；必须确认当前发布工件、哈希和必要依赖均可加载。

### 10.4 错误契约

建议统一为：

```json
{
  "status": "error",
  "error_code": "INVALID_INPUT",
  "message": "请检查标记字段后重新提交。",
  "request_id": "...",
  "details": [
    {"field": "clinical.age", "message": "年龄必须为 18 到 75 岁。"}
  ]
}
```

至少覆盖：

- `INVALID_INPUT`；
- `MODEL_NOT_LOADED`；
- `MODEL_INFERENCE_FAILED`；
- `PYTHON_ENGINE_OFFLINE`；
- `DATABASE_ERROR`；
- `DEVICE_ERROR`；
- `REPORT_ERROR`；
- `WEBVIEW_ERROR`；
- `CONFIG_ERROR`；
- `REQUEST_TIMEOUT`；
- `ARTIFACT_INTEGRITY_ERROR`。

详细堆栈只进入本地技术日志，不能直接回传页面。当前 `/analyze` 会把异常文本拼入响应，迁移时需要修正。

## 11. C# / WinUI 3 宿主职责

第一版宿主只需要做好基础能力，不需要用 XAML 重画 WebUI。

### 11.1 启动顺序

```text
启动 EXE
  -> 初始化 LocalAppData 目录和滚动日志
  -> 读取版本化配置
  -> 初始化/迁移 SQLite
  -> 启动 Python 子进程
  -> 轮询 /api/v1/health 并验证模型版本
  -> 初始化 WebView2
  -> 注入只读 bootstrap 信息
  -> 加载 frontend/dist
```

Python 进程应被纳入 Windows Job Object 或同等父子进程约束，确保主程序异常退出时不会留下后台服务。

### 11.2 退出顺序

```text
阻止新请求
  -> 等待或取消在途分析
  -> 提交/回滚数据库事务
  -> 请求 Python 正常退出
  -> 超时后终止受管子进程
  -> 释放 WebView2
  -> 刷新日志并退出
```

### 11.3 WebView2 消息白名单

第一阶段建议只开放：

- `app.getInfo`；
- `engine.health`；
- `engine.standardize`；
- `engine.analyze`；
- `engine.getOralAdenomaSchema`；
- `engine.analyzeOralAdenoma`；
- `file.openJson`；
- `file.saveJson`；
- `report.savePdf`；
- `report.print`；
- `external.openEvidenceLink`。

不得提供 `shell.execute`、任意文件读写、任意 URL 代理或任意 SQL 执行接口。

## 12. SQLite 方案

数据库由 C# 统一管理，前端 WebUI 和 Python 都不直接连接 SQLite。

### 12.1 第一版表

| 表 | 用途 | 注意事项 |
|---|---|---|
| `patients` | 本地患者主档 | 使用内部 ID，敏感字段单独治理 |
| `samples` | 样本和采集信息 | 与患者一对多 |
| `test_results` | 标准化检测结果 | 保存输入版本和单位 |
| `predictions` | 模型输入摘要、输出和模型版本 | 不把模型分数当诊断结论 |
| `recommendations` | 药学辅助条目和证据版本 | 保留生成时规则版本 |
| `reports` | 报告索引、文件哈希和状态 | 文件内容可独立存储 |
| `users` | 本地用户/角色预留 | 是否启用需产品确认 |
| `audit_logs` | 关键业务操作审计 | 不保存完整敏感原文 |
| `app_settings` | 非敏感应用设置 | 密钥不得明文放入普通设置表 |
| `schema_migrations` | 数据库版本 | 迁移必须可回滚或可备份恢复 |

### 12.2 数据目录

运行数据不应写入安装目录或仓库相对路径。建议使用：

```text
%LOCALAPPDATA%/<Vendor>/<Application>/
  database/
  logs/
  cache/
  engine/
  reports/
```

报告导出到用户选择的位置时，数据库只记录受控路径、哈希、时间和版本。是否需要数据库加密、Windows 用户绑定和医院级备份，必须在 Phase 5 前确定。

## 13. 前端桌面兼容方案

### 13.1 保持一套界面源码

推荐保留 `templates/index.html` 作为当前浏览器入口，同时建立可重复的桌面构建步骤，将同一模板渲染为 `frontend/dist/index.html`。只处理以下 Jinja 变量：

- `app_name`；
- `model_release`；
- `web_model_backend`；
- `internal_oral_adenoma_enabled`；
- 静态 CSS/JS 地址。

桌面运行时由 C# 注入 bootstrap 数据，不手工维护第二份界面 HTML。

### 13.2 传输抽象

新增统一接口，例如：

```ts
interface AppTransport {
  request<TRequest, TResponse>(method: string, payload?: TRequest): Promise<TResponse>
}
```

- 浏览器模式：`HttpTransport` 调用版本化 API；
- 桌面模式：`WebViewTransport` 使用 `window.chrome.webview.postMessage`；
- 页面渲染只依赖领域结果，不依赖端口、Fetch 或 C# 类型。

### 13.3 TypeScript 渐进迁移

先迁移边界，不迁移视觉：

1. 先收敛 API 响应类型和错误；
2. 再将文件与外链操作抽成服务；
3. 再逐步把 `legacy-app.js` 中稳定函数改为 TypeScript；
4. 每一步构建后的 DOM 和 CSS class 保持不变；
5. 不引入 React 作为桌面迁移前置条件。

## 14. 推荐目录增量

不强制移动现有合理目录，只新增清晰边界：

```text
desktop/
  GutOralAxis.Desktop.sln
  GutOralAxis.Desktop/
    App.xaml
    MainWindow.xaml
    Services/
    Bridge/
    Database/
    Reports/
    Logging/
    Models/

ai_engine/
  api/
  schemas/
  service.py
  runtime.py
  errors.py

frontend/
  src/                  # 继续使用现有源码
  desktop/              # bootstrap/transport/build 适配
  dist/                 # 构建产物，不手工编辑

src/                    # 现有算法和运行时逻辑，保持位置
config/releases/        # 发布工件，保持位置
data/pharmacy_knowledge/# 运行时知识库，保持位置
```

`research/`、`experiments/`、`archive/` 和 `ctm_fusion_experiment/` 不应为了匹配理想目录而大规模迁移。

## 15. 文件级迁移顺序

以下顺序保证每阶段仍可运行，并且旧 Web 入口在桌面完成前不会失效。

### Step 0：锁定当前行为

1. 保留当前 `enhanced_app.py` 启动方式；
2. 为核心 API 成功、非法输入、模型不可用和药学降级响应增加契约快照；
3. 为当前页面关键 DOM ID 和 CSS class 建立最小回归清单；
4. 记录现有模型工件哈希和启动能力。

涉及文件：

- `tests/test_app_validation.py`；
- `tests/test_oral_adenoma_bridge.py`；
- 新增 API 契约测试；
- 新增前端 DOM 冒烟测试。

### Step 1：新增 Python Service，不改模型

建议新增：

- `ai_engine/service.py`；
- `ai_engine/runtime.py`；
- `ai_engine/errors.py`；
- `ai_engine/schemas/requests.py`；
- `ai_engine/schemas/responses.py`；
- `ai_engine/api/app.py`；
- `tests/test_ai_engine_api.py`；
- 独立桌面运行时依赖清单。

`service.py` 调用现有 `src.pipeline.run_pipeline()`。先让 Flask 和 FastAPI 并行通过同一 Service，再决定何时退役 Flask API；页面服务可继续保留更久。

### Step 2：拆分推理与保存

修改范围应限制在：

- `enhanced_app.py`：兼容端点使用 Service；
- `src/export_utils.py`：保留 CLI 兼容，不再由纯推理路径强制调用；
- API 响应 schema：移除 Python 强制生成的 `saved_to`，或临时标记为兼容字段。

模型模块不变。

### Step 3：前端传输和系统能力适配

建议新增或调整：

- `frontend/src/transport.ts`；
- `frontend/src/http-transport.ts`；
- `frontend/src/webview-transport.ts`；
- `frontend/src/bootstrap.ts`；
- `frontend/src/api.ts`；
- `frontend/src/types.ts`；
- `frontend/src/oral-adenoma.ts`；
- `frontend/src/legacy-app.js` 中仅替换 API/文件/外链调用；
- 前端桌面静态构建脚本。

不得更改现有 DOM 层级、CSS class、文字信息层级或组件库。

### Step 4：建立最小 WinUI 3 Host

建议新增：

- `desktop/GutOralAxis.Desktop.sln`；
- `desktop/GutOralAxis.Desktop/App.xaml`；
- `desktop/GutOralAxis.Desktop/MainWindow.xaml`；
- `Services/PythonEngineManager.cs`；
- `Services/AiEngineClient.cs`；
- `Bridge/WebViewMessageRouter.cs`；
- `Bridge/BridgeRequest.cs`；
- `Bridge/BridgeResponse.cs`；
- `Services/AppPathService.cs`；
- `Logging/` 下的结构化日志配置；
- 对应单元测试项目。

该阶段只实现窗口、WebView2、Python 生命周期、健康检查和错误页，不实现数据库和设备。

### Step 5：SQLite 和患者数据

新增 C# 数据库层、schema migration、事务测试和备份恢复测试。现有前端 WebUI 只通过白名单消息访问业务服务。

### Step 6：报告与完整业务链

实现结构化报告记录、PDF、打印、保存位置选择、报告索引和审计。Python 只返回报告内容，不直接控制 Windows 打印或 PDF 文件系统。

### Step 7：设备接口

先定义 `IDeviceAdapter` 和标准检测结果 schema，再分别实现 USB/串口适配器。没有真实设备协议前，不把设备代码耦合进现有分析页面。

### Step 8：打包和发布

完成：

- 前端预构建，不向用户分发 Node.js；
- Python Engine 和依赖随安装包交付；
- 只打包运行时工件清单，不打包整个 `outputs/`；
- C# 宿主、WebView2 运行时策略和 Python Engine 一键安装；
- 安装、升级、卸载和数据保留策略；
- 无开发环境干净 Windows 虚拟机验收。

## 16. 安全设计基线

1. Python API 强制绑定 `127.0.0.1`，即使环境变量错误也拒绝非回环地址；
2. 每次启动生成随机令牌，C# 和 Python 之间所有调用均验证；
3. Python API 不信任仅凭“来自 localhost”的请求；
4. WebView2 桥接采用方法白名单和严格 JSON schema；
5. 禁止任意文件路径、任意 URL、任意命令和任意 SQL；
6. 所有 SQL 参数化；
7. 患者原始信息、完整菌群和当前用药不得写入普通日志；
8. 日志记录 request ID、错误码、模块、耗时和版本，不记录完整请求体；
9. 模型和知识库使用版本化 manifest 与 SHA-256；
10. 外部证据链接只允许 `https` 和批准域名，由系统浏览器打开；
11. WebView2 发布版禁用不需要的权限、开发工具和外部导航；
12. 更新包必须签名并验证来源；
13. 数据库备份、导出和删除必须留下审计记录；
14. 医学结果继续明确区分 PFS、研究参考指数和内部口腔研究端点。

## 17. 打包工程重点

### 17.1 Python 分发

建议先采用“安装包内的一目录 Python Engine”进行工程验证，而不是追求单个巨大自解压 Python EXE。PyTorch/PyG/XGBoost 的单文件解压会增加启动时间、杀毒软件扫描和故障定位难度。最终用户仍然只启动主应用，不会接触 Python 目录。

应拆分运行时依赖：

- V8 PFS + 药学基础能力；
- 时间拓扑研究指数能力；
- 可选口腔腺瘤能力；
- 训练/科研依赖不得默认进入生产包。

当前 `requirements.txt` 直接包含全部 `requirements-research.txt`，不适合作为最终桌面运行时依赖清单。

### 17.2 模型工件

当前 V8 JSON 工件约 81 KB，口腔腺瘤 JSON 工件约 43 KB；但时间拓扑路径依赖的现行输出目录约 1.14 GB。Phase 2 必须生成精确 manifest，并通过自动测试证明精简后的发布目录可在无仓库环境完成推理。

### 17.3 CPU/GPU

正式 V8 当前可在 CPU 上运行。桌面第一版建议以 CPU 为默认、GPU 为可选加速能力，不随默认安装包捆绑 CUDA。时间拓扑功能需要单独测量冷启动、峰值内存和单次推理耗时，再决定是否在普通办公电脑上默认预加载。

## 18. 主要风险

| 风险 | 等级 | 影响 | 控制方案 |
|---|---|---|---|
| 把当前前端误判为 React 后重写 | 高 | 视觉回归、周期失控 | 保留现有 DOM/CSS，只加适配层 |
| 默认 V8 隐式加载重型时间拓扑路径 | 高 | 安装包过大、启动慢、内存高 | capability 化、懒加载、运行时工件清单 |
| `/analyze` 推理和文件写入耦合 | 高 | C# 无法统一数据治理 | 先拆为纯分析响应 |
| 没有 SQLite 和迁移机制 | 高 | 患者/报告无法可靠管理 | C# 数据层 + schema version + 备份测试 |
| 生产代码依赖 `research/` 与 `outputs/` | 高 | 安装后路径缺失 | 抽取明确运行时包，不改模型数学逻辑 |
| 模型工件未进入普通 Git 发布 | 高 | CI 安装包可能缺模型 | 受控工件注入、manifest 和哈希门禁 |
| Python 大型依赖打包兼容性 | 高 | 部分 Windows 机器启动失败 | 干净 VM、x64 锁定、依赖探针和冒烟测试 |
| 患者数据日志泄漏 | 高 | 隐私和合规风险 | 默认脱敏、结构化日志、禁记请求体 |
| Jinja 页面转静态资源 | 中 | 两套 HTML 漂移 | 单一模板、构建时生成桌面 HTML |
| 浏览器文件和 `_blank` 行为 | 中 | 文件越权或未受控窗口 | C# 文件服务和外链白名单 |
| 当前前端缺少 UI 自动化测试 | 中 | 小改动造成视觉/交互回归 | DOM 契约 + WebView2 端到端冒烟 |
| 固定端口和多端口重试 | 中 | 冲突或连接错误 | 宿主管理端口、令牌和 readiness |
| WebView2 Runtime/Windows App SDK 差异 | 中 | 部署环境不一致 | 固定版本策略和干净机安装测试 |
| 药品数据许可与离线更新策略未固化 | 中 | 发布和证据过期风险 | 数据来源、许可、版本、复核日期清单 |
| 把内部验证指标解释为临床外部验证 | 高 | 医学表述风险 | 保留当前研究用途和端点边界 |

## 19. 分阶段验收标准

### Phase 1：本分析

- 已确认实际技术栈、页面、API、模型和持久化；
- 已给出 A–E 分类和文件级顺序；
- 未修改现有运行代码、界面或模型。

### Phase 2：Python AI Engine

- `/api/v1/health`, `/standardize`, `/predict`, `/analyze` 可用；
- 旧 Flask 接口回归测试继续通过；
- 同一输入的新旧接口核心模型结果完全一致；
- 非法输入和模型错误返回稳定错误码；
- 推理不再强制写文件；
- 引擎只监听回环地址。

### Phase 3：WebUI 兼容

- 浏览器版视觉不变；
- 桌面静态构建可重复；
- HTTP 和 WebView 两种 transport 行为一致；
- 文件、下载和外链不再依赖未受控浏览器能力；
- 关键 DOM 和截图回归通过。

### Phase 4：Desktop Host

- 双击宿主即可启动 Python 和 WebUI；
- 模型未加载时显示可理解的错误，不进入假就绪；
- 主程序退出后无残留 Python 进程；
- WebView2 不能导航到未批准页面；
- 日志能关联一次完整分析请求。

### Phase 5–6：数据与业务整合

- 患者、样本、检测、预测、建议和报告可以事务化保存；
- 数据库升级和备份恢复通过；
- PDF/打印不需要浏览器或命令行；
- Python 不直接管理 SQLite 和 Windows 文件系统。

### Phase 8：最终发布

- 干净 Windows 机器无需 Python、Node.js 或开发工具；
- 安装后只需启动桌面应用；
- 当前 WebUI 视觉和主要交互基本一致；
- 所有声明的模型能力均通过工件完整性和真实推理冒烟测试；
- 安装、升级、卸载、日志和数据保留行为均有文档和测试。

## 20. 实施前需要锁定的产品决策

这些问题不阻碍 Phase 2，但应在对应阶段前确认：

1. 时间拓扑无病理研究指数是否作为默认安装能力，还是受控可选组件；
2. 内部口腔腺瘤模型是否继续默认关闭；
3. 首发仅支持 x64 CPU，还是同时支持特定 NVIDIA GPU；
4. 使用 MSIX、企业签名安装包或其他受控安装方式；
5. 患者数据库是否需要静态加密、Windows 账户绑定和自动备份；
6. 是否首版引入本地用户角色，还是先依赖 Windows 用户边界；
7. 报告模板、医院信息、签章和打印格式；
8. 药品证据在完全离线环境中的更新和过期提醒策略；
9. 首批设备协议、厂商、VID/PID、串口参数和模拟器要求。

本分析的默认建议是：第一版先完成 CPU/x64、单机 SQLite、现有主分析和药学流程、受控报告导出；设备和多人协作按独立能力增量加入。无病理研究指数如需保留，应完整打包其精简运行时工件，不能静默降级或返回伪结果。

## 21. 最终判断

当前项目不需要重新设计 UI，也不需要先迁移到 React。最稳妥的路线是：

```text
保留当前单页 WebUI
  -> 标准化 Python Service/API
  -> 抽象前端传输和文件能力
  -> 增加 WinUI 3 + WebView2 宿主
  -> 由 C# 接管进程、SQLite、报告、日志和设备
  -> 仅打包经过清单验证的 Python 运行时与模型工件
```

这条路线满足“用户看到的界面尽可能不变”，同时把真正需要桌面化的工程职责移到 C#，并保护当前 AC-ICAM V8、时间拓扑研究路径和药学决策逻辑的可复现性。
