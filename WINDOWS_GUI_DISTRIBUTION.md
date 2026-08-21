# Windows GUI 分发指南

本文件描述如何把 `scripts/build_windows_desktop.ps1` 产出的自包含发布目录，
转成可以交到最终用户手上的三种分发形态。构建本身见 `WINDOWS_GUI_PACKAGING.md`。

## 1. 三种形态怎么选

| 形态 | 产物 | 安装位置 | 提权 | WebView2 | 适用对象 |
| --- | --- | --- | --- | --- | --- |
| 免安装 ZIP | `TCM-Desktop-<版本>-win-x64-<构建号>.zip` | 用户自选 | 否 | 需自备 | 开发者、内部试用、U 盘演示 |
| .exe 安装器 | `GutOralAxis-Desktop-Setup-<版本>-win-x64.exe` | `%LOCALAPPDATA%\Programs`（默认）| 否 | 缺失时自动下载安装 | 公开下载的主推形态 |
| .msi 安装包 | `GutOralAxis-Desktop-<版本>-win-x64.msi` | `Program Files` | 是 | 需信息科统一下发 | 医院/高校批量部署 |

公开分发建议 **.exe 为主、ZIP 为辅、.msi 按需**：

- `.exe` 默认按当前用户安装，不弹 UAC。未签名的安装包一旦要求管理员权限，
  SmartScreen 拦截率明显更高。
- `.msi` 必然提权，但组策略、Intune、SCCM 只认 MSI，机构用户会主动要。
- `ZIP` 给不愿意跑安装程序的人留一条路，也方便你自己回归验证。

## 2. 完整流程

```powershell
# 第一步：构建应用本体（产出发布目录 + ZIP）
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_desktop.ps1 `
  -Python .\.venv-desktop-build\Scripts\python.exe `
  -NuGetMode auto -RunSmoke `
  -AiEngineBundle .\artifacts\ai-engine\<已验证的构建>\goa-ai-engine

# 第二步：生成分发件（自动取最新发布目录）
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_installers.ps1 -Targets Exe,Msi
```

第二步会同时写出 `SHA256SUMS-<版本>.txt`，把 ZIP、EXE、MSI 的校验值放在一起，
随 Release 一并上传，供下载者核对。

## 3. 前置工具

| 目标 | 需要 | 获取 |
| --- | --- | --- |
| .exe | Inno Setup 6 | <https://jrsoftware.org/isdl.php> |
| .exe 中文向导（可选） | `ChineseSimplified.isl` | <https://jrsoftware.org/files/istrans/> |
| .msi | WiX v5 | `dotnet tool install -g wix`，再 `wix extension add -g WixToolset.UI.wixext` |

`build_windows_installers.ps1` 会自动定位 `ISCC.exe`：先查 PATH，再查两个默认安装
目录，最后翻注册表卸载项。装在非默认盘（例如 `D:\Inno Setup 6`）也能找到；
实在不行用 `-InnoSetupPath "D:\Inno Setup 6\ISCC.exe"` 显式指定。

### 3.1 简体中文向导语言文件

**`ChineseSimplified.isl` 不随 Inno Setup 官方安装包提供。** 它属于「非官方翻译」，
官方安装目录 `Languages\` 下只有官方维护的那十几种语言，没有中文。
把这一行写死的后果，就是在没放过该文件的机器上编译直接中断：

```
Error on line 79 in installer.iss: Couldn't open include file
"<Inno Setup 目录>\Languages\ChineseSimplified.isl": 系统找不到指定的文件。
```

现在 `installer.iss` 改为**编译期探测**，按下列顺序找，全都找不到就只出英文向导，
编译不再中断：

1. `ISCC /DChineseIslPath="<绝对路径>"` 显式指定；
2. `desktop/packaging/ChineseSimplified.isl`（随仓库带走，构建可复现，推荐）；
3. `<Inno Setup 安装目录>\Languages\ChineseSimplified.isl`。

要中文向导，就从 <https://jrsoftware.org/files/istrans/> 下载 `ChineseSimplified.isl`，
存到上面第 2 或第 3 个位置。构建脚本会在开头打印当前用的是哪一种：

```
Wizard language   : Simplified Chinese + English
```

或者

```
警告: ChineseSimplified.isl not found - the setup wizard will be ENGLISH ONLY.
```

向导语言只影响安装过程那几屏的文字，**不影响程序本体**——应用界面始终是中文，
安装前的研究用途声明 `installer-notice.txt` 也始终是中文。

### 3.2 260 字符路径上限（MAX_PATH）

ISCC 用的是传统 Win32 文件 API，源文件全路径超过 259 字符就打不开。症状很难认：
先老老实实压缩好几分钟，然后抛出一句既没有行号、也没有文件名的

```
Error in ...\installer.iss: 系统找不到指定的路径。 .
Compile aborted.
```

仓库位于较深目录时，发布目录前缀本身就可能很长，例如：

```
<仓库>\artifacts\windows\TCM-Desktop-0.4.1-win-x64-<构建号>
```

而 AI 引擎包里 `Runtime\Engine\_internal\outputs\` 之下还嵌了好几层训练产物，
最深的两个例子：

| 文件 | 全路径长度 |
| --- | --- |
| `...\split42_three_seed_summary_artifacts\seed_21_fusion_head.pt` | 262 |
| `...\figures\figure_source_data_risk_stratification\consensus_test_predictions_with_groups.csv` | 290 |

`build_windows_installers.ps1` 的处理办法是**缩短前缀而不是动文件**：打包期间用
`subst` 把发布目录映射到一个空闲盘符（从 Z 往前找），编译完在 `finally` 里
`subst /d` 释放。映射后上面两条分别降到 141 和 169 字符。Inno 与 WiX 记录的都是
相对 `SourceDir` 的路径，所以产物和不映射时逐字节一致。

映射成功时开头会打印：

```
Source mapping    : X:\  ->  <仓库>\artifacts\windows\TCM-Desktop-0.4.1-win-x64-...
```

`subst` 成功后脚本还会回查 `X:\GutOralAxis.Desktop.exe` 是否可见——用来识别
「subst 报了成功但映射到了别处」这种情况（含中文的目标路径经控制台代码页传给
`subst.exe` 时有可能走样）。真找不到空闲盘符时脚本只降级告警，不阻断。

> **顺带一个待办**：`Runtime\Engine\_internal\outputs\current_mainline_v2\` 下面
> 那些 `figures\`、`*_history.json`、`figure_source_data_*` 是训练期产物，运行时
> 大概率用不到，却被 PyInstaller 一并收进了引擎包。把它们从引擎打包清单里排掉，
> 既能显著缩小 338 MB 的包体，也能从根上消掉这个路径长度问题。这需要重新构建
> AI 引擎，属于独立的一件事，暂未处理。

## 4. 安装器的既定行为

`.exe`（`desktop/packaging/installer.iss`）：

- 安装前显示 `installer-notice.txt` 的研究用途声明，用户必须翻过这一页。
- 检测 WebView2 运行时（EdgeUpdate 客户端 GUID `{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}`
  的 HKLM/WOW6432Node、HKLM、HKCU 三处），缺失时下载官方 Evergreen 引导程序静默安装。
  下载失败不阻断安装，只提示用户手动补装。
- 默认当前用户安装，用户可在首页切换为全机安装。
- 卸载时默认**保留** `%LOCALAPPDATA%\GutOralAxis`（分析记录与 SQLite 数据库），
  仅在用户取消勾选「保留…」时才删除。

`.msi`（`desktop/packaging/GutOralAxisDesktop.wxs`）：

- 按机器安装到 `Program Files`。
- 同 `UpgradeCode` 自动升级，禁止降级覆盖。
- **不做 WebView2 引导安装**——MSI 内嵌另一个安装程序属于不推荐做法，
  批量部署场景应由信息科统一下发运行时。
- 许可页读取 `desktop/packaging/license.rtf`（MIT 许可 + 研究用途声明，中英双语）。
  该文件由 `desktop/packaging/make_rtf.py` 从仓库根目录的 `LICENSE` 生成，
  全文以 `\uN?` 转义写成纯 ASCII——RTF 里直接塞 UTF-8 字节会在 Windows Installer
  的 RichEdit 控件里变成乱码。改了 `LICENSE` 之后重跑一次即可：

  ```powershell
  python .\desktop\packaging\make_rtf.py
  ```

## 5. 发布前必须处理的三件事

### 5.1 代码签名（公开分发的最大阻碍）

当前产物未签名。未签名 + 体积大 + 从浏览器下载 = SmartScreen 弹
「Windows 已保护你的电脑」，非技术用户到这一步通常直接放弃。

拿到证书后，在 `build_windows_installers.ps1` 之后对**三个产物都签名**
（EXE 安装器、MSI，以及 ZIP 内的 `GutOralAxis.Desktop.exe`——注意主程序要在
打 ZIP 之前签，否则 ZIP 里仍是未签名的）：

```powershell
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 `
  /f <证书>.pfx /p <口令> <目标文件>
```

在拿到证书之前，Release 页面必须附一段说明，告诉用户点「更多信息 →
仍要运行」，并给出 SHA256 让其自行核对。

### 5.2 应用图标（已完成）

全套图标只有**一份**源文件：`desktop/src/GutOralAxis.Desktop/Assets/AppIcon.ico`，
由 `desktop/packaging/make_icon.py` 从 `desktop/packaging/app-icon-source.png`
生成，四处引用它：

| 引用方 | 用途 |
| --- | --- |
| `GutOralAxis.Desktop.csproj` 的 `ApplicationIcon` | 嵌进 exe，任务栏与资源管理器读这个 |
| `MainWindow.xaml.cs` 运行时加载 | 窗口图标 |
| `MainWindow.xaml` 的 `AppIcon.png` | 自定义标题栏左上角那颗 27px 图标 |
| `installer.iss` 的 `SetupIconFile`、`GutOralAxisDesktop.wxs` 的 `ARPPRODUCTICON` | 安装器自身图标、「应用和功能」列表图标、开始菜单快捷方式 |

生成脚本做的三件事，改图标前先读一下：

1. **去白底而不是抠白色。** 白底图每个像素其实是 `p = a*ink + (1-a)*255`。
   脚本先聚类出画面真正用到的两种墨色，再把 `255-p` 投影到 `255-ink` 方向上，
   投影长度即覆盖率、残差最小者即本来的颜色。按亮度阈值硬抠会在抗锯齿边缘留白毛边。
2. **反白 + 深色圆角底板。** 原标志是深蓝灰的，放在 Windows 11 深色任务栏
   （约 `#202020`）上几乎隐形，所以线条反白、青色提亮、垫深色底板。
3. **小尺寸 <256 存 BMP，256 存 PNG。** Pillow 默认把所有帧都写成 PNG；
   PNG 帧是 Vista 才引入的，历史上只在 256 这一档被广泛依赖，小尺寸仍以 BMP 兼容性最好。
   脚本自己写 ICONDIR，不走 Pillow 的写出器。

换图标：替换 `app-icon-source.png`（白底、双色平涂的 PNG），然后

```powershell
python .\desktop\packaging\make_icon.py
```

再重新构建桌面端与安装器。

### 5.3 干净机验收

`WINDOWS_GUI_PACKAGING.md` §10 的最后一项尚未完成：在一台**没有开发环境**的
Windows x64 上装一遍，确认 WebView2 引导、启动、推理、报告、PDF、打印全通。
带 Python/Node 的开发机验不出运行时缺失类问题。

## 6. GitHub Release 建议清单

每个 Release 至少包含：

- `GutOralAxis-Desktop-Setup-<版本>-win-x64.exe`
- `TCM-Desktop-<版本>-win-x64-<构建号>.zip`
- `GutOralAxis-Desktop-<版本>-win-x64.msi`（如面向机构）
- `SHA256SUMS-<版本>.txt`
- 发行说明，必须写明：研究用途边界、系统要求、WebView2 依赖、
  未签名时的 SmartScreen 处理方式，以及 `WINDOWS_GUI_PACKAGING.md` §12
  中尚未完成的能力（自动升级、卸载数据策略、真实设备联调等）。

不要把 `artifacts/` 整个目录、`.test-tmp/`、构建机虚拟环境或 `outputs/`
传上 Release。
