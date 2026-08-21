# Windows GUI 安全说明

> 审阅日期：2026-08-14
> 适用范围：当前仓库中的 WinUI 3 宿主、WebView2 静态 WebUI、C# 桥接、本地 Python AI Engine、SQLite、报告与日志、Windows 发布脚本。
> 文档性质：基于当前源码的工程安全边界说明，不是认证报告。

## 1. 重要声明

当前 Windows GUI 是研究与工程验证软件。本文不声明或暗示其已经满足医疗器械、医院信息系统、网络安全等级保护、HIPAA、GDPR、ISO 13485、IEC 62304、IEC 81001-5-1 或其他法规、标准与认证要求。

当前结果不能直接作为诊断、处方、停药、换药、剂量调整或治疗决策。进入真实患者或临床环境前，必须另行完成组织审批、临床验证、风险管理、隐私影响评估、软件确认、渗透测试、代码签名、部署加固和持续漏洞管理。

本文中的“已实现”只表示代码中存在相应控制，不等于该控制已经经过独立安全审计，也不等于足以抵御具有本机用户权限或管理员权限的攻击者。

## 2. 架构与信任边界

```text
现有静态 WebUI
  | WebView2 message, 协议 v1
  v
WinUI 3 / C# 宿主
  | HTTP, 127.0.0.1, 随机端口, 每次启动随机 token
  v
Python AI Engine

C# 宿主
  +-- %LOCALAPPDATA%\GutOralAxis\Desktop\Data\Database\gut-oral-axis.db
  +-- %LOCALAPPDATA%\GutOralAxis\Desktop\Data\Reports\
  +-- %LOCALAPPDATA%\GutOralAxis\Desktop\Logs\
  +-- %LOCALAPPDATA%\GutOralAxis\Desktop\WebView2\
```

主要信任边界如下：

| 边界 | 当前信任假设 | 不能防御的情况 |
|---|---|---|
| WebUI 到 C# | WebUI 静态资源来自应用发布目录；消息只接受精确 HTTPS 应用虚拟域，且操作受白名单限制 | 发布目录被同机攻击者篡改；同机高权限进程注入或替换应用文件 |
| C# 到 Python | 仅回环地址、随机端口、随机 token；C# 不向页面暴露端口和 token | 同一 Windows 用户或管理员读取进程环境、内存或篡改可执行文件 |
| C# 到本地数据 | 当前 Windows 用户拥有应用数据目录 | 账户被接管、磁盘离线读取、恶意软件、管理员访问 |
| 构建环境到发布物 | 依赖版本、工件白名单和哈希清单由构建脚本生成 | 构建机失陷、清单与文件同时被替换、未签名发布物 |
| 外部证据链接 | 仅允许 HTTPS 主机白名单并交给系统浏览器 | 白名单站点内容变化、域名或终端浏览器本身被攻击 |

## 3. 已实现的安全边界

### 3.1 Windows 宿主与进程生命周期

- [`App.xaml.cs`](../../desktop/src/GutOralAxis.Desktop/App.xaml.cs) 使用本地命名互斥量限制同一用户会话中的重复实例，降低两个进程同时操作本地数据库的概率。
- 应用初始化失败会写入技术日志，并在窗口已建立时显示日志目录。
- [`MainWindow.xaml.cs`](../../desktop/src/GutOralAxis.Desktop/MainWindow.xaml.cs) 在正常关闭时取消生命周期令牌，并只结束本应用创建的 Python Engine 进程树。
- Python Engine 使用隐藏窗口启动，标准输出和错误输出由宿主管理。
- 单实例控制是数据一致性辅助措施，不是用户身份认证或访问控制。

### 3.2 WebView2

当前 [`MainWindow.xaml.cs`](../../desktop/src/GutOralAxis.Desktop/MainWindow.xaml.cs) 已实现：

- 将本地 `WebUI` 映射为 `https://app.gutoralaxis.local/`，资源访问模式为 `DenyCors`。
- 主窗口导航只允许应用虚拟域。
- 新窗口和外部导航不会继续留在 WebView2 中；仅允许预设 HTTPS 主机，并交给系统默认浏览器。
- 默认关闭上下文菜单、浏览器快捷键、状态栏、缩放控制、自动填充和密码保存。
- 默认拒绝 WebView2 权限请求，并取消下载。
- 开发者工具默认关闭，只有环境变量 `GOA_DESKTOP_DEVTOOLS=1` 时开启。
- WebView2 用户数据放在应用数据目录，而不是安装目录。
- `WebMessageReceived` 只接受精确 `https://app.gutoralaxis.local/` HTTPS 默认端口来源；HTTP、自定义端口、用户信息和相似恶意子域均由测试拒绝。
- WebUI 模板启用同源 CSP，禁止远程脚本、对象、frame、媒体和 worker，只允许同源 API 连接。

外部主机白名单当前包括 FDA、DailyMed、RxNav、WHO、CDC、AGA、USPSTF、DOI 与 PMC 等证据站点。白名单只限制主机和 HTTPS，不验证具体页面内容或页面版本。

### 3.3 WebView2 消息桥接

[`BridgeRequestParser.cs`](../../desktop/src/GutOralAxis.Core/Messaging/BridgeRequestParser.cs)、[`BridgeOperationCatalog.cs`](../../desktop/src/GutOralAxis.Core/Messaging/BridgeOperationCatalog.cs) 和 [`BridgeRouter.cs`](../../desktop/src/GutOralAxis.Core/Messaging/BridgeRouter.cs) 已实现：

- 固定消息类型 `goa.request` / `goa.response` 和协议版本 `1`。
- 请求 ID 最长 128 字符，只允许 ASCII 字母、数字、`-`、`_`、`.`。
- WebView2 单条消息最大 2 MiB。
- JSON 最大解析深度为 64，禁止 JSON 注释。
- 只允许显式列出的分析操作和宿主操作，未知操作返回 `OPERATION_NOT_ALLOWED`。
- 当前宿主白名单只包含打开/保存 JSON、保存/列出报告、PDF 导出、打印、读取版本和设备发现。
- 请求和响应通过请求 ID 关联，前端和 Engine 调用均有超时。
- Engine 响应最大 8 MiB；非法 JSON 响应不会直接传给页面。
- 技术异常通常转换为结构化、面向用户的错误，不把内部异常正文直接返回 WebUI。

文件操作还受到以下限制：

- 打开和另存文件使用 Windows 系统选择器，WebUI 不能任意指定本机路径。
- 打开的 JSON 文件最大 2 MiB。
- C# 管理的报告使用安全文件名和根目录约束，拒绝绝对路径和目录穿越。
- 报告写入采用临时文件后移动，并在数据库索引失败时尝试删除未完成文件。

### 3.4 Python AI Engine

[`PythonEngineManager.cs`](../../desktop/src/GutOralAxis.Infrastructure/Engine/PythonEngineManager.cs)、[`ai_engine/__main__.py`](../../ai_engine/__main__.py) 和 [`ai_engine/api/app.py`](../../ai_engine/api/app.py) 已实现：

- C# 固定使用 `127.0.0.1`，Python 入口拒绝非 loopback 地址和 `0.0.0.0`。
- 每次启动由 C# 生成 32 个随机字节，再编码为 64 位十六进制 token。
- token 只通过子进程环境和 `X-GOA-Engine-Token` 请求头传递，不发送给 WebUI。
- Python 使用 `hmac.compare_digest` 校验 token，健康检查和分析接口都要求 token。
- 保留环境变量 `GOA_ENGINE_TOKEN`、`GOA_ENGINE_HOST`、`GOA_ENGINE_PORT` 不能被附加配置覆盖。
- C# HTTP 客户端禁用代理，避免本地 Engine 请求被系统代理转发。
- C# 随机选择回环端口，启动后等待带 token 的健康检查通过。
- Python API 禁用 Swagger、ReDoc 和 OpenAPI 页面，并关闭 Uvicorn access log。
- API 响应带请求 ID 和 `Cache-Control: no-store`。
- 请求体同时按 `Content-Length` 和实际读取字节数限制为 2 MiB，分块请求不能绕过上限。
- API 定义结构化错误，模型异常不会把内部路径和异常正文返回给页面。
- C# 默认不记录 Python stdout/stderr 的正文，只记录“产生了诊断行”。
- 正式构建把 `allow_development_engine_fallback` 固定为 `false`；内置 Engine 缺失时直接失败，不执行环境中的任意 Python。

token 是本机进程间的临时认证，不是面向多用户、恶意本地用户或远程网络的完整认证方案。

### 3.5 SQLite 与报告

[`SqliteDatabase.cs`](../../desktop/src/GutOralAxis.Infrastructure/Database/SqliteDatabase.cs)、[`DatabaseSchema.cs`](../../desktop/src/GutOralAxis.Infrastructure/Database/DatabaseSchema.cs) 和各 Repository 已实现：

- 数据默认写入 `%LOCALAPPDATA%\GutOralAxis\Desktop`，不写入安装目录。
- SQLite 开启外键、WAL、5 秒 busy timeout，并使用 `synchronous=NORMAL`。
- 数据库记录 schema 版本；遇到高于当前程序支持的版本会拒绝启动，防止旧程序误写新数据库。
- Repository 使用参数化 SQL，不拼接用户输入到 SQL。
- 样本、检测结果、预测和建议可在同一事务中保存，失败时回滚。
- JSON 字段由 SQLite `json_valid` 约束。
- 年龄在 Repository 和数据库层限制为 18 至 75 岁或空值。
- 报告保存在受控目录并记录 SHA-256、相对路径和数据库索引。
- 用户、审计日志和应用设置表已经创建；每次桥接操作都会记录操作名、结果状态和时间，不记录请求载荷、患者标识或用药内容。

### 3.6 日志

[`RollingFileLogger.cs`](../../desktop/src/GutOralAxis.Infrastructure/Logging/RollingFileLogger.cs) 已实现：

- 日志按 UTC 日期写入 `desktop-YYYYMMDD.log`。
- 默认保留 30 天，启动时尝试删除过期日志。
- 普通消息转换为单行并限制长度。
- JSON 消息中的患者标识、姓名、电话、邮箱、地址、当前用药、药物过敏和菌群载荷等键会被替换为 `[REDACTED]`。
- 常见 `key=value` 敏感字段也会被替换。
- Python Engine 的原始诊断文本默认不写入 C# 日志。

### 3.7 构建与发布物

[`build_windows_desktop.ps1`](../../scripts/build_windows_desktop.ps1)、[`build_ai_engine_bundle.py`](../../scripts/build_ai_engine_bundle.py) 和相关脚本已实现：

- Node 依赖由 `package-lock.json` 固定；.NET 直接依赖版本集中固定；Python 桌面构建直接依赖有版本约束。
- 可通过经过 NuGet 目录 SHA-512 校验的本地包源恢复 .NET 依赖。
- Python Engine 使用 PyInstaller one-directory 方式打包，终端用户不需要单独安装 Python。
- 只有 [`ai-engine-artifacts.json`](../../desktop/packaging/ai-engine-artifacts.json) 白名单中的模型、配置、药学知识和运行工件进入 Engine 包。
- Engine 包生成 Python 运行时依赖与许可证清单，以及运行时 SHA-256 清单。
- Windows 发布目录生成包含版本、路径、大小和 SHA-256 的 `release-integrity.json`。
- 构建脚本可运行打包后的桌面启动冒烟测试。

这些清单目前是检测材料，不是数字签名，也没有被应用启动流程强制验证。

## 4. 敏感数据处理规则

以下内容应按敏感健康数据处理：患者标识、人口学信息、临床资料、原始菌群数据、药物与过敏信息、模型输入、风险输出、药学辅助建议和报告。

当前运行数据位置：

| 数据 | 默认位置 | 当前保护 |
|---|---|---|
| SQLite | `%LOCALAPPDATA%\GutOralAxis\Desktop\Data\Database` | Windows 用户目录权限、数据库约束；无应用层加密 |
| 报告 | `%LOCALAPPDATA%\GutOralAxis\Desktop\Data\Reports` | 路径限制、文件哈希；无应用层加密 |
| 技术日志 | `%LOCALAPPDATA%\GutOralAxis\Desktop\Logs` | 部分脱敏、30 天保留；无防篡改与加密 |
| WebView2 数据 | `%LOCALAPPDATA%\GutOralAxis\Desktop\WebView2` | 独立用户数据目录；没有自动清理策略 |
| 发布目录 | 管理员选择的位置 | 发布哈希清单；当前无签名、无启动校验 |

部署时必须做到：

1. Windows 账户不得多人共用，设备应启用 BitLocker 或组织批准的磁盘加密。
2. 应用数据目录只授予业务用户和受控管理员，不放在公开共享目录或个人云同步目录。
3. 备份必须进入加密介质或受控备份系统，不通过普通邮件、聊天工具或公共网盘传输。
4. 故障日志在外发前必须人工复核，因为当前异常堆栈不保证完全脱敏。
5. 生产启动环境必须清理开发环境变量，详见第 6 节。

## 5. 明确未覆盖的风险

| 优先级 | 未覆盖项 | 当前影响 | 上线前建议 |
|---|---|---|---|
| 阻断 | 未完成医疗器械和临床用途合规验证 | 不能将输出作为临床决策依据 | 完成预期用途、风险管理、临床验证、软件确认和法规路径评估 |
| 阻断 | SQLite、报告和 WebView2 数据未应用层加密 | 同机账户被接管或磁盘离线读取时可能泄露 | 使用 BitLocker、严格 NTFS ACL；如进入真实患者场景，引入受审计的数据库加密与密钥管理 |
| 阻断 | 没有完整登录、身份认证、RBAC 和会话锁定 | 同一 Windows 账户内无法区分操作者权限 | 接入组织身份体系，建立最小权限角色和会话超时 |
| 高 | 桥接操作已写基础审计，但尚无登录身份、查看事件和防篡改存储 | 无法形成绑定操作者的完整、不可抵赖审计轨迹 | 接入身份体系，并对页面查看、配置、备份和安全事件补充签名审计 |
| 高 | 发布目录中的 WebUI 和二进制仍可被同机高权限攻击者整体替换 | 来源校验无法识别被替换但仍来自应用虚拟域的页面 | 使用 Authenticode 和受保护安装目录，并在启动前验证签名清单 |
| 中 | 当前 CSP 通过 HTML `meta` 生效，不能提供仅响应头支持的 `frame-ancestors` | CSP 已阻止远程脚本和不必要资源，但不是完整的服务端响应头策略 | 如后续改为自定义资源响应，附加等价 CSP 响应头并重新做 WebView2 回归 |
| 高 | 发布哈希清单和 Engine 运行时清单未在启动时强制验证 | 文件与清单可能被一起替换，或篡改直到人工检查才发现 | 启动前校验签名清单；失败时阻止模型加载并记录事件 |
| 高 | EXE、ZIP 和未来安装包未 Authenticode 签名 | SmartScreen 警告，无法可靠确认发布者和完整性 | 使用受控代码签名证书、时间戳服务和双人发布审批 |
| 高 | 没有自动更新、回滚编排和安全公告流程 | 漏洞修复依赖人工分发，版本可能漂移 | 建立签名更新源、分阶段发布、回退包和漏洞响应时限 |
| 高 | C# 日志对 `exception.ToString()` 未做字段级脱敏 | 异常消息可能包含路径或敏感输入 | 写入异常类型和受控错误码，敏感上下文单独脱敏；外发日志前人工审查 |
| 高 | `GOA_DESKTOP_VERBOSE_ENGINE_LOGS=1` 会记录 Engine 原始诊断行 | 诊断内容可能包含内部路径或敏感数据 | 仅在隔离测试机短时启用，收集后立即关闭并受控销毁日志 |
| 中 | 日志按天滚动但没有单文件大小或总容量上限 | 高频诊断行可能占满磁盘并影响数据库写入 | 增加大小轮转、总容量配额、低磁盘告警和写入降级策略 |
| 中 | 回环 HTTP 没有 TLS，token 位于子进程环境 | 同用户高权限进程可观察本机通信或读取 token | 将本机账户边界纳入威胁模型；高敏环境改用命名管道并应用 ACL |
| 中 | 随机端口先保留后释放，再由 Engine 绑定 | 存在很小的本地端口抢占时间窗 | 由宿主持有监听句柄或改用带 ACL 的命名管道 |
| 中 | Engine 异常退出后没有自动重启或熔断策略 | 分析功能保持离线，需重启应用 | 增加有上限的重启、退避和用户确认，不得无限重启 |
| 中 | 报告 SHA-256 只在保存时记录，列出或打开时未重新校验 | 已篡改报告仍可能被展示 | 读取和导出前复核哈希，失败时隔离文件 |
| 中 | 当前没有应用内一致性备份、完整性检查和恢复编排 | 活动 WAL 数据库被直接复制时可能得到不一致备份 | 只执行停机冷备份，并实现经过测试的备份、校验和恢复工作流 |
| 中 | WebView2 用户数据没有清理、保留或取证策略 | 缓存可能长期残留 | 定义版本升级、用户退出、报废和事件响应时的清理策略 |
| 中 | 没有速率限制、资源配额和并发任务队列 | 本机反复请求可能耗尽 CPU、内存或磁盘 | 限制并发、输入复杂度、队列长度和磁盘使用量 |
| 中 | 依赖清单不是完整 SBOM，尚无自动 CVE 扫描 | 已知漏洞可能未及时发现 | 生成标准 SBOM，接入依赖漏洞扫描和发布阻断规则 |
| 中 | 设备层目前只有 `NoDeviceAdapter` | 尚未验证 USB/串口协议、设备身份和数据完整性 | 获取厂商协议后进行设备认证、边界校验、重放和断连测试 |
| 中 | 打包冒烟验证启动、Engine、日志和数据库；独立跨层测试覆盖真实分析 | 尚未覆盖人工打印、恢复演练、全部真实输入和恶意载荷 | 增加目标机人工验收、异常注入、恢复演练和安全回归 |

## 6. 生产环境变量基线

正式发布进程启动前应清除以下开发或测试变量：

| 变量 | 用途 | 生产要求 |
|---|---|---|
| `GOA_DESKTOP_DEVTOOLS` | 开启 WebView2 DevTools | 必须未设置 |
| `GOA_DESKTOP_VERBOSE_ENGINE_LOGS` | 记录 Python 原始诊断文本 | 必须未设置 |
| `GOA_DESKTOP_PYTHON` | 指定开发 Python | 正式包必须未设置 |
| `GOA_DESKTOP_ENGINE_ROOT` | 指定开发 Engine 根目录 | 正式包必须未设置 |
| `GOA_DESKTOP_SMOKE_EXIT` | 启动后自动退出 | 必须未设置 |
| `GOA_DESKTOP_DATA_ROOT` | 覆盖数据目录 | 只允许由受控部署脚本设置到受保护本地路径 |
| `PYTHONPATH` | Python 模块搜索路径 | 正式包必须未设置 |

`desktop-settings.json` 当前控制内部口腔腺瘤功能和开发 Engine 回退。该文件不是授权凭据，也没有签名保护。只有经批准的内部构建才可将 `enable_internal_oral_adenoma` 设为 `true`；发布流程必须将 `allow_development_engine_fallback` 固定为 `false`。

## 7. 上线前最低安全门禁

以下任一项未完成时，不应处理真实患者数据：

- [ ] 完成预期用途和非预期用途评审，并保留“非诊断、非处方”边界。
- [ ] 完成身份认证、角色权限、会话锁定和全面审计接入。
- [ ] 完成数据静态加密、密钥保管、备份加密和恢复演练。
- [x] 已增加 WebView2 精确消息来源校验和同源 CSP。
- [x] 已禁止正式包回退到开发 Python 或任意 Engine 根目录。
- [ ] 对发布物执行启动时签名与完整性校验。
- [ ] 使用 Authenticode 签名并验证时间戳和证书链。
- [ ] 完成依赖 SBOM、CVE 扫描、恶意软件扫描和许可证复核。
- [ ] 完成威胁建模、静态分析、动态分析和独立渗透测试。
- [ ] 完成异常日志脱敏、审计日志防篡改和日志访问控制。
- [ ] 完成数据库损坏、磁盘满、Engine 崩溃、WebView2 失败和断电恢复测试。
- [ ] 完成真实目标 Windows 版本、WebView2 Runtime 和低权限账户测试。
- [ ] 对每个模型版本完成独立验证、变更控制和可追溯发布审批。

## 8. 安全事件处理原则

发现可疑访问、文件哈希不一致、异常导出、恶意软件告警或患者数据疑似泄露时：

1. 停止使用应用进行新的分析，不覆盖或删除现场数据。
2. 正常关闭桌面程序；无法关闭时记录时间、用户和进程信息后再隔离设备。
3. 断开不必要网络连接，保留应用目录、完整应用数据目录和原始日志的只读副本。
4. 对副本计算 SHA-256，并记录采集人、采集时间和存储位置。
5. 日志外发前进行人工敏感信息审查。
6. 由组织安全、隐私和业务负责人判断通知、取证、恢复与监管义务。
7. 只从已知、已验证、已签名的发布物和最近一次经过验证的备份恢复。

详细备份、恢复和排障步骤见 [`WINDOWS_GUI_OPERATIONS.md`](WINDOWS_GUI_OPERATIONS.md)。
