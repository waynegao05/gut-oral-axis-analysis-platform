# Windows GUI 开发指南

## 1. 文档范围

本文说明当前仓库中 Windows GUI 的真实开发方式。迁移工作的首要约束是保留现有 WebUI：

- 页面仍来自 `templates/index.html`。
- 样式仍来自 `static/app.css`。
- TypeScript 入口仍为 `frontend/src/main.ts`，构建结果写入 `static/generated/app.js`。
- 浏览器版入口 `enhanced_app.py` 保留，不因桌面化而移除。
- 桌面版不维护第二套页面；`scripts/build_desktop_web.py` 将同一份 WebUI 渲染为离线资源。
- `src/` 中的模型、特征工程、风险定义和药学逻辑不属于 GUI 重写范围。

当前正式界面是 Jinja 模板、CSS 和原生 JavaScript/TypeScript 的组合，并不是 React 项目。桌面迁移沿用实际技术栈，不为符合概念架构而重写前端。

迁移状态和未完成事项以 `WINDOWS_GUI_TODO.md` 为准。

## 2. 当前架构

```text
同一份 WebUI 源码
templates/index.html + static/app.css + frontend/src/
                |
                | scripts/build_desktop_web.py
                v
frontend/dist/ 离线 WebUI
                |
                | WebView2 虚拟域 https://app.gutoralaxis.local/
                v
desktop/src/GutOralAxis.Desktop (WinUI 3)
                |
                | WebView2 Message，方法白名单
                v
GutOralAxis.Core + GutOralAxis.Infrastructure
                |
                | 带随机令牌的 HTTP，仅 127.0.0.1 随机端口
                v
ai_engine/ (FastAPI + 现有 Python 模型)
```

浏览器模式与桌面模式使用同一份前端业务代码：

- 普通浏览器中，`HttpTransport` 请求现有 Flask 路径。
- WebView2 中，`WebViewTransport` 向 C# 发送 `goa.request` 消息。
- C# 只转发白名单内的模型操作，并接管文件、报告、打印、版本和设备抽象等系统功能。
- 前端看不到 Python 端口和 `GOA_ENGINE_TOKEN`。

## 3. 目录职责

| 路径 | 当前职责 |
| --- | --- |
| `templates/`、`static/` | 正式 WebUI 的模板、视觉样式和浏览器构建结果 |
| `frontend/src/` | 类型化前端逻辑、HTTP/WebView2 双 Transport |
| `frontend/dist/` | 由脚本生成、供 WebView2 离线加载的静态资源 |
| `ai_engine/` | 统一 AI Service/FastAPI 边界，不重写现有模型 |
| `desktop/src/GutOralAxis.Core/` | 消息契约、操作白名单、安全和版本基础类型 |
| `desktop/src/GutOralAxis.Infrastructure/` | Python 生命周期、SQLite、日志、报告和设备抽象 |
| `desktop/src/GutOralAxis.Persistence/` | 持久化程序集入口 |
| `desktop/src/GutOralAxis.Desktop/` | WinUI 3 窗口、WebView2、文件选择、打印和 PDF |
| `desktop/tests/` | Core、Infrastructure、Persistence 冒烟测试程序 |
| `desktop/packaging/` | AI Engine 模型及数据工件白名单 |
| `scripts/` | WebUI、AI Engine、NuGet、完整桌面包和冒烟脚本 |

## 4. 开发环境

当前工程锁定以下桌面构建版本：

- .NET SDK `10.0.302`，见 `desktop/global.json`。
- Windows App SDK `2.3.1`。
- WebView2 SDK `1.0.4078.44`。
- Microsoft.Data.Sqlite `10.0.10`。
- 目标框架 `net10.0-windows10.0.26100.0`。
- 最低 Windows 平台版本 `10.0.19041.0`。
- 当前桌面运行标识为 `win-x64`。

首次准备仓库依赖：

```powershell
npm ci
python -m pip install -r requirements-ai-engine.txt -r requirements-dev.txt
```

运行完整模型时，所选 Python 环境还必须具备模型实际使用的 PyTorch、PyG、XGBoost 等依赖。用于生成独立 Engine 的完整版本集合写在 `requirements-desktop-build.txt`；不要把缺少模型依赖导致的 `CAPABILITY_UNAVAILABLE` 误判为 GUI 故障。

## 5. 构建并检查 WebUI

先检查类型、测试 Transport，再生成浏览器脚本：

```powershell
npm run typecheck
npm run test:frontend
npm run build
```

将同一份 WebUI 生成到 `frontend/dist/`：

```powershell
python scripts/build_desktop_web.py
```

该脚本会：

1. 使用 Jinja 严格模式渲染 `templates/index.html`。
2. 复制 `static/app.css` 和 `static/generated/app.js`。
3. 拒绝未解析的 Jinja 表达式。
4. 生成 `frontend/dist/manifest.json` 及文件 SHA-256。

内部口腔腺瘤功能默认关闭。仅在受控内部构建中显式启用：

```powershell
python scripts/build_desktop_web.py --enable-internal-oral-adenoma
```

浏览器兼容入口仍可独立运行：

```powershell
python enhanced_app.py
```

## 6. 单独调试本地 AI Engine

AI Engine 必须使用至少 32 字符的随机令牌，并且只绑定 loopback 地址：

```powershell
$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$env:GOA_ENGINE_TOKEN = [Convert]::ToHexString($bytes)
python -m ai_engine --host 127.0.0.1 --port 8766
```

不可使用 `--host 0.0.0.0`。`ai_engine.runtime.validate_bind_host()` 会拒绝非 loopback 地址。开发 API 默认关闭 OpenAPI、Swagger、ReDoc 和访问日志，接口请求使用 `X-GOA-Engine-Token`。

当前 Python API 包含：

- `GET /api/v1/health`
- `POST /api/v1/standardize`
- `POST /api/v1/predict`
- `POST /api/v1/analyze`
- `GET /api/v1/oral-adenoma/schema`
- `POST /api/v1/oral-adenoma/analyze`

桌面消息白名单目前代理 `standardize`、`predict`、`analyze` 和两个口腔腺瘤操作；当前页面使用 `standardize` 和 `analyze`，尚未为 `predict` 增加单独的可见入口。

## 7. 编译和运行桌面宿主

先生成 `frontend/dist/`，再还原和编译 WinUI 项目：

```powershell
dotnet restore desktop/src/GutOralAxis.Desktop/GutOralAxis.Desktop.csproj `
  --configfile desktop/NuGet.Config `
  --runtime win-x64

dotnet build desktop/src/GutOralAxis.Desktop/GutOralAxis.Desktop.csproj `
  --configuration Release `
  --runtime win-x64 `
  --no-restore
```

源码开发模式下，宿主会从输出目录逐级查找仓库中的 `ai_engine/__main__.py`。用环境变量指定完整模型 Python 和仓库根目录后启动：

```powershell
$env:GOA_DESKTOP_PYTHON = (Get-Command python).Source
$env:GOA_DESKTOP_ENGINE_ROOT = (Resolve-Path '.').Path
& '.\desktop\src\GutOralAxis.Desktop\bin\Release\net10.0-windows10.0.26100.0\win-x64\GutOralAxis.Desktop.exe'
```

开发时可选环境变量：

| 变量 | 用途 |
| --- | --- |
| `GOA_DESKTOP_PYTHON` | 指定源码模式使用的 Python 可执行文件 |
| `GOA_DESKTOP_ENGINE_ROOT` | 指定包含 `ai_engine/` 的工作目录 |
| `GOA_DESKTOP_DATA_ROOT` | 将数据库、日志、报告和 WebView2 数据重定向到测试目录 |
| `GOA_DESKTOP_DEVTOOLS=1` | 仅开发时启用 WebView2 开发者工具 |
| `GOA_DESKTOP_VERBOSE_ENGINE_LOGS=1` | 记录 Engine 诊断文本；默认只记录发生过诊断输出 |
| `GOA_DESKTOP_SMOKE_EXIT=1` | GUI 就绪后自动退出，供冒烟测试使用 |

宿主启动时会随机保留一个 `127.0.0.1` 端口，生成 32 字节随机 Engine Token，隐藏启动自己拥有的 Python 子进程，健康检查通过后加载业务链路。关闭窗口时只终止该宿主创建的 Engine 进程。

## 8. WebView2 与 C# 边界

WebView2 从本地目录映射的虚拟域加载：

```text
https://app.gutoralaxis.local/index.html
```

当前安全约束包括：

- 只允许应用虚拟域内导航。
- 外部链接必须为 HTTPS 且主机位于证据来源白名单，随后交给系统浏览器。
- 默认关闭上下文菜单、浏览器快捷键、状态栏、缩放控制、自动填充和密码保存。
- 默认拒绝 WebView2 权限请求和下载。
- WebView2 消息上限为 2 MiB，默认前端等待时间为 120 秒。
- 消息必须使用版本 1 的 `goa.request`/`goa.response` 信封和关联请求 ID。

模型消息白名单：

```text
standardize
analyze
oralAdenoma.schema
oralAdenoma.analyze
```

Windows 宿主消息白名单：

```text
file.openJson
file.saveJson
report.save
report.list
report.exportPdf
report.print
app.getVersion
device.discover
```

没有列入白名单的操作会返回 `OPERATION_NOT_ALLOWED`。设备层当前使用 `NoDeviceAdapter`；在取得真实 USB/串口协议前，不应伪造设备驱动。

## 9. 本地数据与配置

默认数据根目录：

```text
%LOCALAPPDATA%\GutOralAxis\Desktop\
```

主要子目录：

```text
Data\Database\gut-oral-axis.db
Data\Reports\
Logs\
Runtime\
WebView2\
```

患者数据不会写入安装目录。开发和自动测试应通过 `GOA_DESKTOP_DATA_ROOT` 使用隔离目录。

桌面功能开关位于 `desktop/src/GutOralAxis.Desktop/desktop-settings.json`。源码配置中的 `allow_development_engine_fallback=true` 仅用于本仓库开发运行；它允许内置 Engine 不存在时使用显式配置的开发 Python。`scripts/build_windows_desktop.ps1` 会在正式发布目录强制写成 `false`，因此发布包不会回退到环境中的任意 Python。五类版本位于同目录的 `version-manifest.json`：Application、Frontend、AI Engine、Model 和 Database Schema。不要在没有同步版本清单的情况下发布模型或数据库 schema 变更。

## 10. 回归检查

Python 边界与桌面构建脚本测试：

```powershell
python -m pytest `
  tests/test_ai_engine_service.py `
  tests/test_ai_engine_api.py `
  tests/test_desktop_web_build.py `
  tests/test_desktop_packaging.py `
  tests/test_fetch_nuget_packages.py
```

前端检查：

```powershell
npm run typecheck
npm run test:frontend
npm run build
python scripts/build_desktop_web.py
```

C# 分层冒烟程序：

```powershell
dotnet run --project desktop/tests/GutOralAxis.Core.SmokeTests/GutOralAxis.Core.SmokeTests.csproj --configuration Release
dotnet run --project desktop/tests/GutOralAxis.Persistence.SmokeTests/GutOralAxis.Persistence.SmokeTests.csproj --configuration Release
dotnet run --project desktop/tests/GutOralAxis.Infrastructure.SmokeTests/GutOralAxis.Infrastructure.SmokeTests.csproj --configuration Release
```

最后还应检查浏览器模式仍能启动、桌面页面视觉未发生非必要变化，以及 `src/` 模型核心没有因 GUI 迁移产生意外差异。完整发布冒烟流程见 `WINDOWS_GUI_PACKAGING.md`。

## 11. 常见问题

### 桌面提示缺少 WebUI

先执行 `npm run build` 和 `python scripts/build_desktop_web.py`。桌面项目只打包已经存在的 `frontend/dist/`。

### Engine 启动后立即退出

确认 `GOA_DESKTOP_PYTHON` 指向包含完整模型依赖的解释器，并检查 `%LOCALAPPDATA%\GutOralAxis\Desktop\Logs\`。默认日志会抑制 Python 诊断正文，调试时可临时设置 `GOA_DESKTOP_VERBOSE_ENGINE_LOGS=1`。

### 手动启动 Engine 报 Token 错误

必须先生成并设置长度不少于 32 字符的 `GOA_ENGINE_TOKEN`。桌面宿主会自动生成令牌，正常桌面使用不需要手工设置。

### NuGet 无法联网

开发构建可改用 `desktop/NuGet.Local.Config`，但前提是 `desktop/.local-nuget/` 已有完整且校验过的包缓存。准备方式见打包文档，不能用不完整缓存冒充离线构建成功。
