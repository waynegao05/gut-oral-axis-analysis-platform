# Windows GUI 打包指南

## 1. 当前交付边界

当前仓库实现的是 `win-x64` 自包含便携发布流程：

- 同一份现有 WebUI 被构建为离线资源，不重新设计页面。
- WinUI 3/.NET 桌面宿主通过 `dotnet publish --self-contained true` 发布。
- Python AI Engine 由 PyInstaller `--onedir` 生成 `goa-ai-engine.exe`，随桌面程序一起分发。
- C# 在启动时自动运行 Engine；最终用户不需要安装 Python、Node.js，也不需要手动启动 FastAPI或浏览器。
- Engine 只监听随机选择的 `127.0.0.1` 端口，并使用每次启动生成的随机令牌。
- 发布目录和 ZIP 都写入 `artifacts/windows/`。

当前脚本尚未生成 MSI/MSIX 安装器，也未执行代码签名、自动升级或卸载数据迁移。这些仍是 `WINDOWS_GUI_TODO.md` 中的发布阶段事项。因此，现阶段产物应称为“自包含便携包”，不能标记为已签名正式安装包。

WebView2 SDK 包不等于 WebView2 浏览器运行时。目标机需要可用的 Microsoft Edge WebView2 Runtime；当前打包脚本没有附带 Fixed Version Runtime 或 Evergreen Bootstrapper。

## 2. 输入和版本来源

完整打包入口：

```text
scripts/build_windows_desktop.ps1
```

脚本读取：

- `desktop/src/GutOralAxis.Desktop/version-manifest.json`：应用、前端、Engine、模型和数据库版本。
- `desktop/src/GutOralAxis.Desktop/desktop-settings.json`：桌面功能开关默认值。
- `desktop/Directory.Packages.props`：Windows App SDK、WebView2 和 SQLite 版本。
- `desktop/packaging/ai-engine-artifacts.json`：允许进入 Engine 包的模型、配置和数据工件白名单。
- `requirements-desktop-build.txt`：PyInstaller 和完整模型运行依赖。

发布前先确认版本清单与待发布模型一致。打包脚本使用 UTC 时间生成唯一目录名：

```text
TCM-Desktop-<application-version>-win-x64-<yyyyMMddTHHmmssZ>
```

同名目录或 ZIP 已存在时，脚本会停止，不会覆盖旧产物。

## 3. 构建机准备

构建必须在 Windows x64 上进行。需要：

- `desktop/global.json` 指定的 .NET SDK `10.0.302` 或其允许的最新补丁。
- Node.js 和由 `package-lock.json` 锁定的前端依赖。
- 可创建独立构建环境的 Python。
- 用于在线准备依赖的网络，或已经准备好的 npm、Python 和 NuGet 本地缓存。

建议使用隔离 Python 环境：

```powershell
python -m venv .venv-desktop-build
.\.venv-desktop-build\Scripts\python.exe -m pip install --upgrade pip
.\.venv-desktop-build\Scripts\python.exe -m pip install -r requirements-desktop-build.txt
npm ci
```

`requirements-desktop-build.txt` 当前包含 CPU 版 PyTorch、PyG、XGBoost、PyYAML 和 PyInstaller。不要直接打包一个混有 Jupyter、Qt 或无关科研工具的日常 Python 环境；PyInstaller 脚本虽然排除了常见 GUI/Jupyter 模块，隔离环境仍能减少非预期依赖和包体积。

如果仓库路径中的非 ASCII 字符导致虚拟环境启动器异常，应将构建环境放到 ASCII 路径，或先将仓库映射到临时盘符；不要修改模型或打包脚本来绕过路径问题。

## 4. NuGet 模式

`scripts/build_windows_desktop.ps1` 支持三种 `-NuGetMode`：

| 模式 | 行为 |
| --- | --- |
| `online` | 使用 `desktop/NuGet.Config` 从 NuGet.org 还原 |
| `verified-local` | 仅使用 `desktop/.local-nuget/` 和 `desktop/NuGet.Local.Config` |
| `auto` | 存在 `microsoft.windowsappsdk.2.3.1.nupkg` 时选本地，否则选在线 |

首次准备经过 SHA-512 校验的本地 NuGet 源，可执行仓库现有下载器：

```powershell
python scripts/fetch_nuget_packages.py `
  --output-dir desktop/.local-nuget `
  Microsoft.WindowsAppSDK=2.3.1 `
  Microsoft.Web.WebView2=1.0.4078.44 `
  Microsoft.Data.Sqlite=10.0.10 `
  Microsoft.NETCore.App.Runtime.win-x64=10.0.10 `
  Microsoft.WindowsDesktop.App.Runtime.win-x64=10.0.10 `
  Microsoft.AspNetCore.App.Runtime.win-x64=10.0.10 `
  Microsoft.Windows.SDK.NET.Ref=10.0.26100.57
```

下载器会解析依赖、核对 NuGet 注册目录发布的 SHA-512，并写入 `packages-manifest.json`。`verified-local` 只保证 NuGet 不联网；完整离线构建还要求 `node_modules/` 和 Python 构建环境已经在断网前准备完毕。

## 5. 单独构建 AI Engine

先检查 PyInstaller 计划和工件白名单，不生成文件：

```powershell
.\.venv-desktop-build\Scripts\python.exe scripts/build_ai_engine_bundle.py `
  --output-dir artifacts/ai-engine/preview `
  --work-dir .test-tmp/pyinstaller-preview `
  --plan-only
```

正式生成 Engine 时必须使用尚不存在的输出目录：

```powershell
.\.venv-desktop-build\Scripts\python.exe scripts/build_ai_engine_bundle.py `
  --output-dir artifacts/ai-engine/20260814-build01 `
  --work-dir .test-tmp/pyinstaller-20260814-build01
```

预期入口：

```text
artifacts/ai-engine/20260814-build01/goa-ai-engine/goa-ai-engine.exe
```

Engine 目录同时包含：

- `ai-engine-artifacts.json`：本次使用的工件白名单副本。
- `python-dependencies.json`：构建环境中实际收集到的 Python 依赖和许可证信息。
- `runtime-integrity.json`：Engine 目录内文件的 SHA-256、大小和总量。

白名单当前只允许 V8/口腔腺瘤发布配置、时间拓扑配置、药学规则与知识库、Topology V6 数据，以及当前主线的时间拓扑和 full-risk 工件进入包。必需工件缺失时构建会失败，不会生成一个静默降级的 Engine。

## 6. 一键生成桌面便携包

在线 NuGet 构建并执行打包后冒烟：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_desktop.ps1 `
  -Python .\.venv-desktop-build\Scripts\python.exe `
  -NuGetMode online `
  -RunSmoke
```

使用完整本地 NuGet 缓存：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_desktop.ps1 `
  -Python .\.venv-desktop-build\Scripts\python.exe `
  -NuGetMode verified-local `
  -RunSmoke
```

复用已经验证过的 Engine，避免再次运行 PyInstaller：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_desktop.ps1 `
  -Python .\.venv-desktop-build\Scripts\python.exe `
  -NuGetMode verified-local `
  -AiEngineBundle .\artifacts\ai-engine\20260814-build01\goa-ai-engine `
  -RunSmoke
```

仅受控内部版本可追加：

```text
-EnableInternalOralAdenoma
```

该开关会同时写入桌面静态 WebUI 和发布目录的 `desktop-settings.json`。普通发布不传此开关，内部口腔腺瘤功能保持关闭。无论是否启用内部入口，发布脚本都会把 `allow_development_engine_fallback` 写成 `false`；缺少随包 Engine 时应用必须失败，不得调用开发机或目标机环境中的任意 Python。

## 7. 一键脚本实际步骤

`scripts/build_windows_desktop.ps1` 按以下顺序执行，任一步失败都会终止：

1. TypeScript 类型检查。
2. 前端 Transport 测试。
3. 使用 esbuild 生成压缩后的 `static/generated/app.js`。
4. 由 `scripts/build_desktop_web.py` 生成离线 `frontend/dist/`。
5. 构建或校验 `goa-ai-engine.exe`。
6. 根据 `-NuGetMode` 选择 NuGet 配置并执行 `dotnet restore`。
7. 对 WinUI 工程执行 `Release`、`win-x64`、`self-contained` 发布。
8. 将 Engine 复制到发布目录的 `Runtime/Engine/`。
9. 写入 `desktop-settings.json`，并强制关闭开发 Engine 回退。
10. 生成整个发布目录的 `release-integrity.json`。
11. 用 `Compress-Archive` 生成同名 ZIP。
12. 指定 `-RunSmoke` 时，隐藏启动发布后的真实 GUI 并检查启动、Engine、退出日志和 SQLite 初始化。

脚本不会运行全部 Python 和 C# 回归测试。发布前应先完成 `WINDOWS_GUI_DEVELOPMENT.md` 中的分层测试，再运行打包脚本。

## 8. 发布产物

成功后脚本输出 JSON，包含发布目录、ZIP、应用版本、模型版本和实际 NuGet 模式。目录结构关键部分如下：

```text
artifacts/windows/
  TCM-Desktop-<version>-win-x64-<build-id>/
    GutOralAxis.Desktop.exe
    WebUI/
      index.html
      assets/
    Runtime/
      Engine/
        goa-ai-engine.exe
        _internal/
        ai-engine-artifacts.json
        python-dependencies.json
        runtime-integrity.json
    desktop-settings.json
    version-manifest.json
    release-integrity.json
  TCM-Desktop-<version>-win-x64-<build-id>.zip
```

`.NET` 和 Windows App SDK 的自包含文件也位于发布目录，未在上图逐项展开。不要只复制 `GutOralAxis.Desktop.exe`；应完整交付整个目录或 ZIP。

## 9. 打包后冒烟

一键脚本的 `-RunSmoke` 会调用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_windows_desktop.ps1 `
  -ApplicationDirectory '.\artifacts\windows\TCM-Desktop-<version>-win-x64-<build-id>'
```

冒烟脚本会：

- 确认桌面 EXE、捆绑 Engine EXE、项目 PRI 和两个 XBF 资源都存在。
- 清空源码模式的 Python/Engine 环境变量，防止误用开发环境。
- 将运行数据隔离到 `.test-tmp/packaged-desktop-<id>/`。
- 隐藏启动真实 WinUI 应用。
- 要求进程以退出码 0 结束，且日志中没有初始化或 Engine 启动失败。
- 要求日志出现应用就绪、Engine 启动和有序退出事件。
- 要求 SQLite 数据库完成初始化。

它验证的是“自包含包可以启动并管理捆绑 Engine”。此外，C# Infrastructure 冒烟会直接启动同一打包 Engine 并执行真实 `analyze`。这些自动门禁仍不替代完整临床输入的人工验收、PDF、打印和真实设备测试。

## 10. 发布前检查表

- [ ] `version-manifest.json` 与本次模型和应用版本一致。
- [ ] `desktop-settings.json` 的内部功能开关符合发布对象。
- [ ] `ai-engine-artifacts.json` 只包含审核通过的必需工件。
- [ ] Python API、前端和三个 C# 冒烟项目均通过。
- [ ] 现有浏览器入口 `python enhanced_app.py` 仍可运行。
- [ ] 桌面 WebUI 与现有 WebUI 的布局、样式和主要交互没有非必要变化。
- [ ] 打包使用隔离 Python 环境，不混入无关开发包。
- [ ] `release-integrity.json` 和 Engine 的 `runtime-integrity.json` 已生成。
- [ ] `-RunSmoke` 通过，且测试没有回退到源码 Python。
- [ ] 在干净的 Windows x64 终端机上验证 WebView2 Runtime、启动、推理、报告、PDF 和打印。
- [ ] 正式对外发布前完成代码签名、安装器、升级和卸载数据保留策略。

## 11. 目标机运行与数据边界

解压完整 ZIP 后运行：

```text
GutOralAxis.Desktop.exe
```

用户不需要执行 npm、Python、FastAPI 或数据库命令。应用数据默认写入：

```text
%LOCALAPPDATA%\GutOralAxis\Desktop\
```

升级或替换便携包时，不应删除该目录。当前尚无安装器级卸载选项，因此患者数据备份、保留和恢复必须在正式部署流程中单独确认。

发布包内不得包含真实患者输入、开发日志、`.test-tmp/`、构建机虚拟环境、NuGet 缓存或仓库完整 `outputs/`。模型和数据只能通过 `ai-engine-artifacts.json` 白名单进入 Engine。

## 12. 当前尚未完成的发布能力

以下能力不能因已有 ZIP 而视为完成：

- MSI/MSIX 或其他标准安装包。
- 发布证书签名和签名验证。
- 自动升级、回滚和版本兼容策略。
- 卸载时患者数据保留/清除选项。
- WebView2 Runtime 的引导安装或 Fixed Version Runtime 随包分发。
- 真实 USB/串口设备驱动和硬件联调。
- 干净终端机上的完整业务闭环验收。

这些项目继续按照 `WINDOWS_GUI_TODO.md` 管理；完成前应在发行说明中明确限制。
