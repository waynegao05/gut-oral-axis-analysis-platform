# Windows GUI 运维手册

> 审阅日期：2026-08-14
> 适用范围：当前仓库 Windows GUI 的开发构建、便携发布、启动停止、数据备份恢复、日志检查和故障排查。
> 当前定位：研究与工程验证软件，不是经过医疗器械合规确认的临床系统。

## 1. 当前交付形态

当前仓库提供：

- WinUI 3 + WebView2 桌面宿主源码；
- 保留现有视觉和交互的离线静态 WebUI 构建；
- 由 C# 自动启动和停止的本地 Python AI Engine；
- SQLite、结构化报告、滚动日志和设备空实现；
- PyInstaller Engine 打包脚本；
- self-contained `win-x64` 桌面发布和 ZIP 生成脚本；
- 启动冒烟测试与若干分层测试。

当前没有正式安装包、自动更新器、代码签名、静默部署方案或生产级回滚服务。最终用户包仍依赖目标机器存在可用的 Microsoft Edge WebView2 Runtime。

## 2. 版本与关键目录

版本来源是 [`version-manifest.json`](../../desktop/src/GutOralAxis.Desktop/version-manifest.json)，分别记录 Application、Frontend、AI Engine、Model 和 Database Schema。构建前必须核对该文件与实际模型发布一致。

默认运行数据根目录：

```text
%LOCALAPPDATA%\GutOralAxis\Desktop
```

目录内容：

| 路径 | 内容 | 备份要求 |
|---|---|---|
| `Data\Database\gut-oral-axis.db` | SQLite 主数据库 | 必须备份 |
| `Data\Database\gut-oral-axis.db-wal` | SQLite WAL，运行时可能存在 | 应用关闭后与数据库一起备份 |
| `Data\Database\gut-oral-axis.db-shm` | SQLite 共享内存文件，运行时可能存在 | 应用关闭后与数据库一起备份 |
| `Data\Reports\` | 按年月保存的结构化报告 | 必须备份 |
| `Logs\` | 每日日志，默认保留 30 天 | 按组织策略备份或归档 |
| `WebView2\` | WebView2 用户数据和缓存 | 一般不作为业务恢复必需项，但事件取证时应保留 |
| `Runtime\` | 预留运行数据目录 | 发现内容时一并审阅 |

安装或解压目录包含：

```text
GutOralAxis.Desktop.exe
WebUI\
Runtime\Engine\goa-ai-engine.exe
desktop-settings.json
version-manifest.json
release-integrity.json
```

安装目录与数据目录必须分开管理。升级应用目录不应覆盖 `%LOCALAPPDATA%` 中的业务数据。

## 3. 构建前提

推荐在专用 Windows 构建机和全新 PowerShell 进程中构建。最低要求：

- Windows x64；
- `.NET SDK 10.0.302`，由 [`desktop/global.json`](../../desktop/global.json) 约束；
- Node.js，以及通过 `npm ci` 安装的锁定依赖；
- 能运行 `requirements-desktop-build.txt` 的 Python 构建环境；
- 足够磁盘空间，PyTorch、NuGet 缓存、PyInstaller 工作目录和发布物会占用数 GiB；
- 在线 NuGet，或已完整准备的 `desktop/.local-nuget`；
- 用于最终生产发布时，还需要组织批准的签名证书和独立扫描流程，当前脚本未包含。

建议先检查工作区：

```powershell
git status --short --branch
dotnet --version
node --version
python --version
```

构建会重建 `static/generated/app.js` 和 `frontend/dist/`，并在被忽略的 `artifacts/`、`.test-tmp/`、`desktop/.packages/` 中产生文件。应在可追溯的干净分支或专用工作副本中执行，不要把患者数据放入仓库。

## 4. 准备构建依赖

### 4.1 Node 与 Python

```powershell
npm ci
python -m venv .venv-desktop-build
& .\.venv-desktop-build\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-desktop-build\Scripts\python.exe -m pip install -r requirements-desktop-build.txt
```

如果当前含中文路径导致虚拟环境启动器异常，可使用已验证的 Conda Python，或在单独 PowerShell 中临时映射短盘符后构建。映射只用于构建，不应写入发布配置：

```powershell
subst G: (Get-Location).Path
Set-Location G:\
# 在此执行依赖安装和构建
Set-Location $env:SystemDrive\
subst G: /D
```

### 4.2 在线 NuGet

构建命令使用 `-NuGetMode online` 时，.NET 从 `https://api.nuget.org/v3/index.json` 恢复允许映射的 Microsoft 和 SQLite 包。

### 4.3 经哈希验证的本地 NuGet

离线或可重复构建前，可在联网准备机执行：

```powershell
& .\.venv-desktop-build\Scripts\python.exe scripts\fetch_nuget_packages.py `
  Microsoft.WindowsAppSDK=2.3.1 `
  Microsoft.Web.WebView2=1.0.4078.44 `
  Microsoft.Data.Sqlite=10.0.10 `
  Microsoft.NETCore.App.Runtime.win-x64=10.0.10 `
  Microsoft.WindowsDesktop.App.Runtime.win-x64=10.0.10 `
  Microsoft.AspNetCore.App.Runtime.win-x64=10.0.10 `
  Microsoft.Windows.SDK.NET.Ref=10.0.26100.57 `
  --output-dir desktop\.local-nuget
```

下载脚本会按 NuGet 目录元数据校验 SHA-512，并递归获取依赖。应将整个 `desktop/.local-nuget` 作为受控构建缓存保存，不能只复制三个顶层包。

## 5. 构建 Windows 便携包

### 5.1 完整构建

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_windows_desktop.ps1 `
  -Python .\.venv-desktop-build\Scripts\python.exe `
  -NuGetMode verified-local `
  -RuntimeIdentifier win-x64 `
  -RunSmoke
```

没有本地 NuGet 缓存时，将 `verified-local` 改为 `online`。`auto` 会在发现指定 Windows App SDK 包时选择本地缓存，否则使用在线源。

如已单独生成并验证 Engine 包，可复用它：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_windows_desktop.ps1 `
  -Python .\.venv-desktop-build\Scripts\python.exe `
  -NuGetMode verified-local `
  -AiEngineBundle artifacts\ai-engine\<build-id>\goa-ai-engine `
  -RunSmoke
```

内部口腔腺瘤功能默认关闭。只有审批通过的内部构建才可附加 `-EnableInternalOralAdenoma`，该开关不是权限控制。

### 5.2 预期输出

成功后生成：

```text
artifacts\windows\TCM-Desktop-<application-version>-win-x64-<UTC-build-id>\
artifacts\windows\TCM-Desktop-<application-version>-win-x64-<UTC-build-id>.zip
```

发布目录内的 `release-integrity.json` 记录除自身之外所有文件的相对路径、大小和 SHA-256。Engine 目录内还有 `runtime-integrity.json`、`python-dependencies.json` 和 `ai-engine-artifacts.json`。

不要把生成这些清单等同于签名。当前应用启动时不会自动阻止哈希不一致的发布物。

## 6. 发布前验证

### 6.1 分层测试

在构建环境运行：

```powershell
npm run typecheck
npm run test:frontend

& .\.venv-desktop-build\Scripts\python.exe -m pytest `
  tests\test_ai_engine_api.py `
  tests\test_ai_engine_service.py `
  tests\test_desktop_packaging.py `
  tests\test_desktop_web_build.py `
  tests\test_fetch_nuget_packages.py

dotnet run --project desktop\tests\GutOralAxis.Core.SmokeTests

$env:GOA_TEST_ROOT = Join-Path $env:TEMP "goa-persistence-$([guid]::NewGuid().ToString('N'))"
dotnet run --project desktop\tests\GutOralAxis.Persistence.SmokeTests

$env:GOA_TEST_ROOT = Join-Path $env:TEMP "goa-infrastructure-$([guid]::NewGuid().ToString('N'))"
dotnet run --project desktop\tests\GutOralAxis.Infrastructure.SmokeTests
```

需要执行真实开发 Engine 闭环时：

```powershell
$env:GOA_TEST_ENGINE = "1"
$env:GOA_DESKTOP_PYTHON = (Resolve-Path .\.venv-desktop-build\Scripts\python.exe).Path
$env:GOA_DESKTOP_ENGINE_ROOT = (Get-Location).Path
$env:GOA_TEST_ROOT = Join-Path $env:TEMP "goa-engine-$([guid]::NewGuid().ToString('N'))"
dotnet run --project desktop\tests\GutOralAxis.Infrastructure.SmokeTests
```

完成后关闭该 PowerShell，避免开发环境变量污染正式启动。

### 6.2 打包后启动冒烟

`build_windows_desktop.ps1 -RunSmoke` 会调用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\smoke_windows_desktop.ps1 `
  -ApplicationDirectory artifacts\windows\<release-directory>
```

当前冒烟会验证：

- 桌面 EXE 和打包 Engine EXE 存在；
- 应用能启动并自动退出；
- 日志出现 `engine.started`、`application.ready` 和正常关闭事件；
- SQLite 数据库成功初始化。

当前冒烟不等于完整临床流程测试，也不验证 PDF、打印、备份恢复、所有模型输入或安全攻击场景。

### 6.3 手工校验发布哈希

在分发前和解压后各执行一次：

```powershell
$root = (Resolve-Path "artifacts\windows\<release-directory>").Path
$manifest = Get-Content -Raw -LiteralPath (Join-Path $root "release-integrity.json") | ConvertFrom-Json
$failures = @()
foreach ($entry in $manifest.files) {
    $path = Join-Path $root ($entry.path -replace '/', '\')
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $failures += "MISSING $($entry.path)"
        continue
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne ([string]$entry.sha256).ToLowerInvariant()) {
        $failures += "MISMATCH $($entry.path)"
    }
}
if ($failures.Count -gt 0) {
    $failures
    throw "Release integrity verification failed."
}
"Release integrity verification passed: $($manifest.file_count) files"
```

哈希通过只能说明文件与同目录清单一致。正式发布还必须校验 Authenticode 签名和可信发布渠道，而当前仓库尚未实现签名流程。

## 7. 安装与首次启动

当前为便携目录，不是安装器：

1. 在受控终端上验证 ZIP 来源和哈希。
2. 解压到普通用户不能随意替换的本地目录，例如由管理员管理的 `C:\Program Files\GutOralAxis` 或组织批准路径。
3. 确认 `Runtime\Engine\goa-ai-engine.exe`、`WebUI\index.html`、`version-manifest.json` 和 `desktop-settings.json` 存在。
4. 确认 Microsoft Edge WebView2 Runtime 已安装且符合组织补丁策略。
5. 清除第 8 节所列开发变量。
6. 启动 `GutOralAxis.Desktop.exe`。
7. 等待启动遮罩消失，再进行分析。
8. 检查最新日志包含 `database.initialized`、`engine.started` 和 `application.ready`。

正常启动顺序：创建 AppData 目录、初始化 SQLite、启动 Engine、带 token 健康检查、初始化 WebView2、加载本地 WebUI。

Engine 启动失败时，宿主当前仍可能加载 WebUI；分析请求随后返回 `PYTHON_ENGINE_OFFLINE`。因此不能只看到页面就判断模型已经可用，必须检查日志中的 `engine.started`。

## 8. 运行配置

### 8.1 发布配置文件

`desktop-settings.json` 当前支持：

```json
{
  "enable_internal_oral_adenoma": false,
  "allow_development_engine_fallback": false
}
```

`enable_internal_oral_adenoma` 控制受限的内部口腔腺瘤研究入口。`allow_development_engine_fallback` 只供源码开发：设为 `true` 时，内置 Engine 不存在才允许使用开发 Python；正式构建脚本始终把它写成 `false`，内置 Engine 缺失时直接报错。

正式环境不要手工编辑该文件来绕过功能审批或启用开发回退。应由构建流程生成，并与发布版本一同留档。

### 8.2 环境变量

| 变量 | 合法用途 | 运维要求 |
|---|---|---|
| `GOA_DESKTOP_DATA_ROOT` | 测试或受控部署的数据根目录覆盖 | 只能指向本机受保护目录；改变后会形成另一套独立数据 |
| `GOA_DESKTOP_PYTHON` | 开发模式指定 Python | 正式包清除 |
| `GOA_DESKTOP_ENGINE_ROOT` | 开发模式指定仓库根目录 | 正式包清除 |
| `GOA_DESKTOP_VERBOSE_ENGINE_LOGS` | 临时诊断 Engine | 默认清除；启用时按敏感日志处理 |
| `GOA_DESKTOP_DEVTOOLS` | 开发调试 WebView2 | 正式包清除 |
| `GOA_DESKTOP_SMOKE_EXIT` | 自动化启动测试 | 正式包清除 |
| `PYTHONPATH` | 开发模块定位 | 正式包清除 |

可用以下命令检查当前 PowerShell：

```powershell
Get-ChildItem Env: | Where-Object Name -Match '^(GOA_|PYTHONPATH$)'
```

## 9. 正常停止与进程检查

优先通过窗口关闭按钮正常退出。关闭流程会取消请求、停止本应用创建的 Engine 进程树、刷新日志并释放单实例互斥量。

关闭后检查残留进程：

```powershell
Get-Process -Name GutOralAxis.Desktop,goa-ai-engine -ErrorAction SilentlyContinue
```

只有在正常关闭失败并记录现场后，才使用任务管理器或 `Stop-Process`。强制终止期间不要复制正在写入的 SQLite 数据库。

## 10. 冷备份

当前没有应用内备份功能。最可靠的现有方式是在应用完全关闭后备份整个数据根目录。

### 10.1 备份步骤

```powershell
$source = Join-Path $env:LOCALAPPDATA "GutOralAxis\Desktop"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$destination = "E:\GutOralAxis-Backups\Desktop-$stamp"

if (Get-Process -Name GutOralAxis.Desktop,goa-ai-engine -ErrorAction SilentlyContinue) {
    throw "Close GutOralAxis Desktop and its Engine before backup."
}
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "Data root does not exist: $source"
}

New-Item -ItemType Directory -Path $destination -Force | Out-Null
robocopy $source $destination /E /COPY:DAT /DCOPY:DAT /R:2 /W:1 /XJ
$copyExit = $LASTEXITCODE
if ($copyExit -ge 8) {
    throw "Backup copy failed with robocopy exit code $copyExit."
}

Get-ChildItem -LiteralPath $destination -Recurse -File |
    Get-FileHash -Algorithm SHA256 |
    Export-Csv -NoTypeInformation -Encoding UTF8 "$destination.sha256.csv"
```

将 `E:` 替换为组织批准的加密备份位置。备份目录和哈希清单应一起保存，但访问权限分开审阅。哈希清单不是加密，也不会隐藏患者信息。

### 10.2 建议频率

- 研究验证阶段：每次导入重要队列、模型版本切换和批量分析前后备份。
- 有持续数据录入时：至少每日冷备份，并定期复制到独立故障域。
- 至少每季度执行一次恢复演练；真实业务环境应按 RPO/RTO 风险评估提高频率。
- 保留周期由组织的数据政策决定，不要直接沿用日志的 30 天保留期。

## 11. 非破坏恢复

恢复前必须保存当前目录，而不是直接覆盖或删除：

```powershell
$root = Join-Path $env:LOCALAPPDATA "GutOralAxis\Desktop"
$backup = "E:\GutOralAxis-Backups\Desktop-<timestamp>"
$quarantine = "$root.pre-restore-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

if (Get-Process -Name GutOralAxis.Desktop,goa-ai-engine -ErrorAction SilentlyContinue) {
    throw "Close GutOralAxis Desktop and its Engine before restore."
}
if (-not (Test-Path -LiteralPath $backup -PathType Container)) {
    throw "Backup directory does not exist: $backup"
}
if (Test-Path -LiteralPath $root) {
    Move-Item -LiteralPath $root -Destination $quarantine
}
New-Item -ItemType Directory -Path $root -Force | Out-Null
robocopy $backup $root /E /COPY:DAT /DCOPY:DAT /R:2 /W:1 /XJ
$copyExit = $LASTEXITCODE
if ($copyExit -ge 8) {
    throw "Restore copy failed with robocopy exit code $copyExit. Original data remains at $quarantine"
}
```

恢复后：

1. 启动与备份 schema 兼容的应用版本。
2. 检查日志中的 `database.initialized` 和 `application.ready`。
3. 检查患者、报告索引和关键报告数量。
4. 运行一条不写入真实患者数据的批准测试案例。
5. 确认恢复成功前，不删除 `.pre-restore-*` 目录。

当前 schema 版本为 1。程序会拒绝高于自身支持版本的数据库。应用降级可能无法读取未来 schema，因此回滚时应使用升级前的数据库副本，不能让旧版本直接打开已经升级的唯一生产数据库。

## 12. 日志运维

日志默认位置：

```text
%LOCALAPPDATA%\GutOralAxis\Desktop\Logs\desktop-YYYYMMDD.log
```

常见事件：

| 事件 | 含义 | 运维动作 |
|---|---|---|
| `application.start` | 宿主开始启动 | 与后续 ready/failed 配对检查 |
| `database.initialized` | SQLite schema 可用 | 不代表业务数据完整性已验证 |
| `engine.started` | token 健康检查通过 | 分析前必须出现 |
| `engine.start_failed` | Engine 无法启动 | 检查包、杀毒隔离、路径和详细异常 |
| `engine.exited` | Engine 进程退出 | 当前不会自动恢复，保存工作并重启应用 |
| `engine.request_failed` | 本地 Engine 通信中断 | 检查 Engine 是否退出及资源压力 |
| `webview.navigation_blocked` | 阻止了非白名单外链 | 核对页面是否被篡改或链接是否需要审批 |
| `report.saved` | 结构化报告已保存 | 可按 report ID 追踪 |
| `host.operation_failed` | 文件、打印或报告等宿主操作失败 | 结合时间和用户操作排查 |
| `application.ready` | WebUI 已加载 | 仍应确认同次启动有 `engine.started` |
| `application.stop` | 正常退出完成 | 备份前应看到该事件 |

日志消息做了部分脱敏，但异常堆栈和临时启用的 Engine 原始诊断可能包含路径或敏感信息。不要直接把完整日志上传到公开工单或聊天工具。

查看最近日志：

```powershell
$logRoot = Join-Path $env:LOCALAPPDATA "GutOralAxis\Desktop\Logs"
Get-ChildItem -LiteralPath $logRoot -Filter "desktop-*.log" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 |
    Get-Content -Tail 200
```

## 13. 故障排查

| 现象 | 可能原因 | 处理步骤 |
|---|---|---|
| 双击后立即退出且无新窗口 | 已有实例持有单实例互斥量 | 检查现有窗口和进程；正常关闭旧实例后重试 |
| 启动遮罩停留或显示初始化失败 | 数据目录权限、数据库、WebUI 或 WebView2 初始化失败 | 查看最新日志；确认磁盘空间、目录权限、`WebUI\index.html` 和 WebView2 Runtime |
| 页面可见但分析提示 Engine 离线 | Engine 启动失败或启动后退出 | 查找 `engine.start_failed` / `engine.exited`；确认 `Runtime\Engine\goa-ai-engine.exe` 未被隔离；重启应用 |
| Engine 启动超时 | 首次模型加载慢、内存不足、杀毒扫描或工件缺失 | 等待不超过 3 分钟；检查内存、事件日志和发布工件；不要无限提高超时掩盖问题 |
| 分析在约 2 分钟后超时 | 单次 Engine 请求超过宿主时限 | 记录请求 ID、输入规模和资源占用；重启后用批准测试数据复现 |
| 返回 `REQUEST_TOO_LARGE` 或页面提示数据过大 | WebView2 请求超过 2 MiB | 不要拆分成不完整医学记录；检查是否误上传大文件或重复数组 |
| 返回 `OPERATION_NOT_ALLOWED` | 前端和宿主版本不匹配，或页面尝试未知操作 | 核对五类版本和发布哈希；不要临时放宽白名单 |
| 返回 `AUTHENTICATION_FAILED` | 直接调用 Engine、token 不匹配或进程混用 | 不支持用户直接调用 Engine；关闭残留进程并由宿主重新启动 |
| 数据库锁定或写入超时 | 另一个进程、备份软件或安全软件占用文件 | 确认只有一个应用实例；停止对活动数据库的复制；等待 5 秒后重试 |
| 数据库 schema 高于支持版本 | 使用旧应用打开了新数据库 | 停止操作；改用匹配版本，或从升级前冷备份恢复 |
| 报告保存失败 | 磁盘满、目录权限、路径或数据库索引失败 | 检查 `Data\Reports`、数据库和磁盘；不要手工伪造 reports 表记录 |
| PDF 导出失败 | WebView2 打印服务、目标路径或权限异常 | 换本地可写路径；检查 WebView2 Runtime；保留结构化 JSON 报告 |
| 打印对话框无设备 | Windows 无可用打印机或策略禁用 | 检查系统打印服务；不要把网页截图当作正式可追溯报告 |
| 外部证据链接打不开 | 主机不在 HTTPS 白名单或系统浏览器策略阻止 | 核对日志和目标主机；通过代码审阅更新白名单，不在现场关闭限制 |
| SmartScreen 或杀毒告警 | 当前发布物未签名，PyInstaller 包可能触发启发式检测 | 停止分发；核对哈希并提交组织安全团队；不要要求用户关闭安全软件 |
| 设备列表为空 | 当前只有 `NoDeviceAdapter` | 属于预期行为；未获得真实协议前不要伪造 USB/串口数据 |

## 14. 发布升级与回退

当前没有自动更新器。每次升级应执行：

1. 冻结并记录五类版本和 Git 提交。
2. 运行全部分层测试、打包冒烟、哈希验证和组织安全扫描。
3. 对现有数据目录执行冷备份和恢复抽检。
4. 将新发布目录部署到独立版本路径，不直接覆盖旧目录。
5. 在测试账户和测试数据根目录完成首次启动与业务冒烟。
6. 再切换正式快捷方式或启动入口。
7. 保留上一发布目录、上一 ZIP、哈希清单和升级前数据备份。

需要回退时：

- 先停止新版本并保存日志和当前数据目录副本；
- 如果数据库 schema 未变化，可在副本上验证旧版读取；
- 如果 schema 已变化，只恢复升级前备份，不要让旧版修改唯一的新数据库；
- 将启动入口切回上一经过验证的发布目录；
- 记录回退原因、影响范围、数据时间点和验证结果。

## 15. 数据库与报告注意事项

- WAL 模式下运行时可能存在 `db`、`db-wal` 和 `db-shm` 三个文件；不要只复制主 `db`。
- 当前应用没有“数据库完整性检查”或“从备份恢复”按钮，必须按受控流程操作。
- `AnalysisRepository` 支持原子写入，但原 WebUI 没有患者主档或患者标识流程；当前桌面桥接主要完成分析转发和显式 `report.save`。在患者身份、授权和界面流程确认前，系统不会虚构患者记录，也不能假定每次分析已自动写入全部业务表。
- 桥接分析、文件、报告、打印、版本和设备操作会写不含载荷的基础审计；当前仍没有登录身份、页面查看事件和不可抵赖审计存储。
- 报告保存时记录 SHA-256，当前读取和列表流程不会自动拒绝被篡改文件。
- 手工移动报告文件会破坏数据库相对路径索引。迁移时应备份和恢复整个 `Data` 目录。

## 16. 日常巡检清单

每日或每次使用前：

- [ ] 确认应用来自批准目录，版本与计划一致。
- [ ] 确认没有开发环境变量残留。
- [ ] 确认磁盘空间充足，数据目录不在公开共享或个人云同步路径。
- [ ] 启动后确认同次日志包含 `engine.started` 和 `application.ready`。
- [ ] 确认没有持续出现 `engine.exited`、数据库错误或导航阻断异常峰值。

每次发布：

- [ ] 记录 Git 提交、五类版本、构建机和构建时间。
- [ ] 使用锁定依赖并保存依赖清单。
- [ ] 完成测试、打包冒烟和发布哈希复核。
- [ ] 完成恶意软件、依赖漏洞和许可证审查。
- [ ] 完成代码签名；当前脚本未实现，未签名包不得作为正式临床发布。
- [ ] 完成升级前备份与恢复抽检。

每月或按组织策略：

- [ ] 检查 Windows、WebView2 Runtime 和安全软件补丁状态。
- [ ] 检查日志保留和备份成功率。
- [ ] 随机抽取备份执行恢复演练。
- [ ] 复核用户目录 ACL、共享路径和异常导出记录。
- [ ] 复核依赖漏洞和模型版本是否有撤回或更新要求。

## 17. 已知运维缺口

以下内容尚未由当前代码或脚本提供：

- MSI/MSIX 或其他正式安装器；
- Authenticode 签名与时间戳；
- 自动更新、灰度发布和自动回滚；
- 应用内备份、恢复、数据库完整性校验和迁移预演；
- SQLite 与报告加密、集中密钥管理；
- 完整用户认证、RBAC、会话锁定和全业务审计；
- 启动时强制校验发布签名和 SHA-256 清单；
- 集中监控、告警、崩溃转储收集和隐私安全上传；
- WebView2 Runtime 离线引导安装与版本合规检查；
- 真实 USB/串口设备驱动和设备安全验证；
- 干净终端、低权限账户、断电、磁盘满和恶意载荷的完整验收报告；
- 医疗器械或临床系统合规材料。

安全边界和上线阻断项见 [`WINDOWS_GUI_SECURITY.md`](WINDOWS_GUI_SECURITY.md)。
