# Windows GUI 迁移 To Do

> 更新日期：2026-08-14
> 原则：保留现有 WebUI、模型与 Flask 入口；采用增量迁移；每个阶段通过门禁后再进入下一阶段。
> 状态：`[x]` 已完成，`[~]` 进行中，`[ ]` 待完成，`[!]` 受外部条件阻断。
> 当前工程状态：桌面主链路、便携包与本机自动化验证已完成；安装签名、真实设备、真实患者身份流程和干净目标机验收仍受外部条件限制。

## 0. 不可破坏项

- [x] 记录迁移前架构与风险：`GUI_MIGRATION_ANALYSIS.md`。
- [x] 确认正式 WebUI 是 Jinja + CSS + 原生 JavaScript/TypeScript，不因文档中的 React 示例而重写。
- [x] 确认正式模型入口、临床适用范围和无病理研究指数路由。
- [x] 保留 `enhanced_app.py` 浏览器入口。
- [x] 保留 `templates/`、`static/app.css` 的视觉与 DOM 结构。
- [x] GUI 迁移不修改 `src/` 模型公式、权重、特征工程或风险定义。

## 1. Python AI Engine

- [x] 增加独立 `ai_engine/` Service/API 层。
- [x] 实现 `health`、`standardize`、`predict`、`analyze` 和口腔腺瘤接口。
- [x] API 仅允许 loopback 地址，拒绝 `0.0.0.0`。
- [x] 增加每次启动随机令牌、结构化错误、请求 ID 和载荷限制。
- [x] 将推理与报告文件写入解耦。
- [x] 完成真实 Uvicorn 进程冒烟测试。
- [x] 完成 Python 回归测试，当前门禁记录见 `AI_ENGINE_MIGRATION_GATE.md`。

## 2. WebUI 双 Transport

- [x] 定义浏览器 HTTP 与 WebView2 Message 共用的 TypeScript 传输契约。
- [x] 将主分析的直接 `fetch()` 收敛到 Transport，不改变结果渲染函数。
- [x] 将口腔腺瘤请求收敛到相同 Transport。
- [x] 增加请求 ID、超时、载荷限制和桌面响应校验。
- [x] 桌面模式不向页面暴露 Python 端口或 Engine Token。
- [x] 保留浏览器模式对现有 Flask 路径的兼容。
- [x] 为 HTTP/WebView2 Transport 增加自动化测试。

**门禁：** `npm run typecheck`、前端单元测试和生产构建全部通过；现有关键 DOM ID 与 CSS 哈希未因 Transport 修改而变化。

## 3. 桌面静态 WebUI

- [x] 从同一份 `templates/index.html` 生成桌面 `frontend/dist/`，不维护第二套页面。
- [x] 构建时注入应用名、模型版本和功能开关。
- [x] 复制当前 CSS 与构建后的 JS，禁止外部网络字体或 CDN 依赖。
- [x] 检查桌面 HTML 不残留 Jinja 表达式。
- [x] 检查关键 DOM ID、文本层级和样式引用与 Web 版一致。

**门禁：** 静态页面可离线加载，且 DOM/CSS 结构回归测试通过。

## 4. WinUI 3 + WebView2 宿主

- [x] 创建 `desktop/` 解决方案和 WinUI 3 主程序。
- [x] 锁定 .NET、Windows App SDK、WebView2 和 SQLite 版本。
- [x] 初始化主窗口、独立 WebView2 用户数据目录和虚拟静态域。
- [x] 仅允许应用静态域导航；批准的 HTTPS 外链交系统浏览器。
- [x] 仅接受精确应用来源的 WebView2 Message，并以测试覆盖恶意来源。
- [x] 在原 WebUI 模板启用同源 CSP，不改变页面布局或视觉。
- [x] 发布配置关闭开发者工具、默认上下文菜单、下载和不必要权限。
- [x] 实现单实例、正常启动和有序退出。
- [x] 修复 `dotnet publish` 遗漏 XBF/PRI 的问题，并纳入构建必需文件检查。

**本机条件：** 使用 .NET 10 SDK 和仓库本地、哈希校验后的 NuGet 缓存；不依赖 Visual Studio 模板。

**门禁：** WinUI 工程可还原、编译并启动，WebView2 能离线显示原 WebUI。

## 5. C# 桥接与 Python 生命周期

- [x] 建立 WebView2 消息信封、方法白名单和 schema/大小校验。
- [x] 实现请求/响应关联、超时、取消和结构化错误。
- [x] C# 隐藏启动 Python Engine，随机选择 loopback 端口并传递随机令牌。
- [x] 健康检查通过后开放分析；离线时返回 `PYTHON_ENGINE_OFFLINE`。
- [x] Engine 异常退出时返回可理解错误并记录技术日志。
- [x] 应用退出时只终止本应用创建的 Python 子进程。
- [x] C# 代理 `standardize`、`predict`、`analyze` 和口腔腺瘤接口。
- [x] 直接启动打包 Engine 完成真实分析和进程回收测试。

**门禁：** 从 WebView2 表单到现有 Python 模型形成真实闭环，且浏览器 Flask 模式仍可运行。

## 6. SQLite、配置、日志、文件和报告

- [x] 建立 AppData 目录策略，不向安装目录写患者数据。
- [x] 建立 SQLite schema 版本与幂等迁移器。
- [x] 建立 `patients`、`samples`、`test_results`、`predictions`、`recommendations`、`reports`、`users`、`audit_logs`、`app_settings` 表。
- [x] 所有 SQL 参数化；事务、JSON 约束和外键默认启用。
- [x] 患者标识、原始输入和完整药物信息不写入普通日志。
- [x] 建立按日滚动文件日志、启动/退出/错误记录和敏感字段脱敏。
- [x] C# 接管打开/保存 JSON、文件选择和路径校验。
- [x] C# 保存结构化报告、记录 SHA-256 并维护报告索引。
- [x] 建立打印与 PDF 导出服务，并由当前 WebUI 提供预览。
- [x] 对每次 WebView2 桥接操作写入不含患者载荷的基础审计记录。
- [!] 原 WebUI 不含患者主档/身份流程；在标识、授权和产品界面确定前，不自动虚构患者并写入全部业务表。

**门禁：** 数据库迁移幂等；CRUD/事务/脱敏/路径遍历测试通过；报告可保存、索引和重新打开。

## 7. 设备、版本和发布

- [x] 定义 USB/串口设备适配器接口和明确的无设备实现。
- [!] 获得真实设备协议后才能实现具体驱动；当前不伪造设备通信。
- [x] 建立 Application、Frontend、AI Engine、Model、Database Schema 五类版本清单。
- [x] 生成 Python 运行时与模型工件白名单、SHA-256、依赖和许可证清单。
- [x] 建立前端、AI Engine、桌面宿主和便携包的一键构建脚本。
- [x] 生成自包含 `Application.exe` 目录和 ZIP，目标用户无需手动启动 Python、Node.js 或命令行服务。
- [!] MSI/MSIX、签名、自动升级和卸载数据选项需要发布证书及产品部署决策。
- [!] 仍需在一台无开发环境的干净 Windows x64 终端完成独立验收。

**外部门禁：** 具体设备驱动需要厂商协议；正式安装包签名需要发布证书。

## 8. 完整验证

- [~] Python API 与模型全量回归：346 项通过；另有 1 项既存 V6 数据/归档哈希不一致，未由迁移改动引起。
- [x] TypeScript 类型、Transport、生产构建与桌面静态资源测试。
- [x] C# Core、Infrastructure、Persistence 测试。
- [x] WinUI/WebView2 真实启动冒烟。
- [x] 浏览器 WebUI/Flask 路由回归包含在 Python 全量测试中。
- [x] 打包 Engine 的健康、标准化、真实分析、关联与退出闭环测试。
- [x] 最终便携目录逐文件复算 SHA-256：4,574/4,574 项一致。
- [x] 最终 ZIP 可读取，SHA-256 已记录于 `WINDOWS_GUI_BUILD_REPORT.md`。
- [x] 非法年龄、负菌群值、NaN/Infinity、超大载荷、恶意来源和路径遍历测试。
- [~] 已覆盖 Engine 离线、事务回滚、报告路径失败和 WebUI 缺失；数据库锁竞争、打印失败和恢复演练仍需目标机人工验证。
- [x] 核对 `src/` 模型核心无 GUI 迁移改动。
- [x] 核对现有 DOM/CSS 和用户可见布局无非必要变化。

## 9. 交付文档

- [x] `WINDOWS_GUI_DEVELOPMENT.md`：开发、调试和目录说明。
- [x] `WINDOWS_GUI_PACKAGING.md`：离线构建、便携交付、安装和签名边界。
- [x] `WINDOWS_GUI_SECURITY.md`：信任边界、消息白名单和敏感数据处理。
- [x] `WINDOWS_GUI_OPERATIONS.md`：日志、备份、恢复和故障排查。
- [x] `WINDOWS_GUI_BUILD_REPORT.md`：本次产物、测试、哈希与上线边界。
- [x] 更新本清单，区分已完成项、部分完成项与外部依赖项。

## 剩余外部 To Do

1. 在干净 Windows x64 终端执行人工启动、分析、PDF 和打印验收。
2. 产品侧确认患者身份流程、安装器形式、签名证书和升级/卸载策略。
3. 接入组织身份、权限、加密、备份恢复和安全审计制度。
4. 获得厂商 USB/串口协议后，再进入真实设备驱动阶段。
5. 完成预期用途、临床、隐私、安全和法规评审后，再决定是否进入真实患者环境。
