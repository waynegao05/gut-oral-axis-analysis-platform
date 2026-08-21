# Windows GUI 构建与验证报告

> 构建日期：2026-08-14
> 应用版本：0.1.0
> 前端版本：2.0.0
> AI Engine 版本：1.0.0
> 模型版本：ac_icam_real_outcome_pfs_v8
> 数据库 Schema：1

## 1. 交付产物

- 便携目录：`artifacts/windows/TCM-Desktop-0.1.0-win-x64-20260814T111531Z/`
- ZIP：`artifacts/windows/TCM-Desktop-0.1.0-win-x64-20260814T111531Z.zip`
- ZIP 大小：344,367,947 字节
- ZIP SHA-256：`97a40bed30715005e40dab7c0b43c14034f151a259f5a6396c52e15b15183996`

这是自包含 Windows x64 便携包。用户不需要手工启动 Python、Node.js、FastAPI 或浏览器；目标机仍需可用的 Microsoft Edge WebView2 Runtime。

上述目录和 ZIP 是本机构建验证产物，受 `.gitignore` 管理。本次 GitHub 预发布只发布源码，不上传便携包附件，因为完整运行包包含保持在本地的模型工件。

## 2. 自动验证结果

| 层级 | 结果 |
|---|---|
| WinUI Release 编译 | 通过，0 个警告、0 个错误 |
| C# Core | 8/8 通过 |
| C# Persistence | 8/8 通过 |
| C# Infrastructure + 发布包 Engine | 6/6 通过 |
| Python API 定向回归 | 34/34 通过 |
| Python 全量回归 | 346 通过，1 个既存 V6 归档哈希断言失败 |
| WebUI Transport | 6/6 通过 |
| TypeScript 类型与生产构建 | 通过 |
| TypeScript 客户端 | 4/4 通过，打包清单 15 个文件 |
| PowerShell 脚本语法 | 14/14 通过 |
| 自包含 GUI 启动 | 通过，进程退出码 0 |
| 发布包 Engine 真实分析 | 通过，覆盖健康、标准化、分析、请求关联和退出 |
| 发布目录完整性 | 4,574/4,574 个清单文件哈希一致，共 840,751,346 字节 |
| ZIP 可读性 | 通过，共 4,597 个目录/文件条目 |

唯一未通过的 Python 测试是 `test_v6_archive_matches_preserved_source_files`。它反映当前 V6 数据与其归档副本原本就不一致；GUI 迁移未修改这两处数据，本次没有覆盖或删除任一版本。

## 3. 已验证的关键边界

- 原 WebUI 的模板、布局、CSS 和主要交互继续复用。
- 浏览器 Flask 模式与 Windows WebView2 模式共用同一前端和模型入口。
- Python API 仅监听 `127.0.0.1`，每次启动使用随机 token。
- 正式包设置 `allow_development_engine_fallback=false`，缺少内置 Engine 时不调用环境中的 Python。
- WebView2 只接受精确应用来源的消息，并对宿主操作使用白名单。
- WebUI 启用同源 CSP，不加载远程脚本或不必要的嵌入资源。
- SQLite 使用参数化 SQL、外键、事务和 schema 版本。
- 桥接审计只记录操作名与结果状态，不记录患者载荷。
- 发布包未包含仓库的 `tests/`、开发虚拟环境、Node 模块、Git 元数据或运行期患者数据库；第三方运行库自身可能包含其包内测试资源。
- Engine 内的两组 `outputs/current_mainline_v2` 内容是模型工件白名单明确要求的正式权重与汇总，不是仓库输出目录的无选择复制。

## 4. 尚未满足的上线条件

以下项目需要外部资料、证书、组织制度或独立环境，不能由当前仓库单方面宣告完成：

- MSI/MSIX 安装器形式、Authenticode 代码签名、时间戳和升级/卸载策略。
- 一台无 Python、Node.js、.NET SDK 和源码的干净 Windows x64 终端验收。
- 真实 USB/串口设备厂商协议、设备身份和异常场景测试。
- 患者身份、授权、登录、RBAC、会话锁定和完整防篡改审计。
- 数据静态加密、密钥管理、备份恢复演练和低磁盘/断电测试。
- 医疗器械、隐私、临床用途与独立安全合规评审。
- 打印机和 PDF 在目标机构环境中的人工验收。

完成上述条件前，当前包应作为研究和工程验证版本使用，不能表述为已获准处理真实患者数据的正式临床软件。
