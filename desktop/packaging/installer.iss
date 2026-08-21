; ============================================================================
;  Gut-Oral Axis Desktop — Inno Setup 安装器脚本（产出 .exe）
;
;  用法（不要直接双击本文件，由 scripts/build_windows_installers.ps1 调用）：
;    ISCC.exe /DSourceDir="<发布目录绝对路径>" /DAppVersion="0.4.1" installer.iss
;
;  设计取舍：
;  - 默认按「当前用户」安装到 %LOCALAPPDATA%\Programs，不触发 UAC。
;    未签名的安装包一旦要求管理员权限，SmartScreen 拦截率会明显上升。
;    用户仍可在首页选择「为所有用户安装」，那种情况下才提权。
;  - 卸载默认保留 %LOCALAPPDATA%\GutOralAxis 下的分析记录与数据库，
;    只在用户明确勾选时才删除。这是医疗研究工具，不能默默清数据。
;  - 安装前检测 WebView2 运行时，缺失时下载官方 Evergreen 引导程序静默安装。
; ============================================================================

#ifndef SourceDir
  #error 必须通过 /DSourceDir=... 传入已构建好的发布目录
#endif

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#ifndef OutputDir
  #define OutputDir "..\..\artifacts\windows"
#endif

#ifndef OutputBaseFilename
  #define OutputBaseFilename "GutOralAxis-Desktop-Setup-" + AppVersion + "-win-x64"
#endif

#define AppName "Gut-Oral Axis Desktop"
#define AppNameZh "肠-口轴分析平台"
#define AppPublisher "Gut-Oral Axis Research"
#define AppExeName "GutOralAxis.Desktop.exe"
; WebView2 Evergreen 引导程序官方短链（微软维护，勿改）
#define WebView2BootstrapUrl "https://go.microsoft.com/fwlink/p/?LinkId=2124703"

; ============================================================================
;  简体中文向导语言的探测
;
;  ChineseSimplified.isl 不随 Inno Setup 官方安装包提供——它属于「非官方翻译」，
;  官方安装目录下的 Languages\ 里只有 Default.isl 之外的十几种官方语言，
;  中文需要自己从 https://jrsoftware.org/files/istrans/ 下载后放进去。
;  硬写死这一行的后果就是：换一台没放过该文件的机器，编译直接中断。
;
;  所以这里在编译期按顺序探测，全都找不到就只出英文向导，编译不中断：
;    1) /DChineseIslPath="<绝对路径>"  显式指定（优先级最高）
;    2) 本脚本同目录下的 ChineseSimplified.isl（把它随仓库带走，构建可复现）
;    3) <Inno Setup 安装目录>\Languages\ChineseSimplified.isl
; ============================================================================
; 两点写法上的讲究，都是为了「探测本身绝不会把编译打断」：
;   - 一律先 #ifdef 再取值。预处理器变量没定义就直接引用会报未声明标识符。
;   - 一律自己补 "\"。CompilerPath / SourcePath 带不带尾部反斜杠不必纠结，
;     路径中间出现连续反斜杠 Windows 会自行归一，两种情况都能打开。
#ifndef ChineseIslPath
  #ifdef SourcePath
    #if FileExists(SourcePath + "\ChineseSimplified.isl")
      #define ChineseIslPath SourcePath + "\ChineseSimplified.isl"
    #endif
  #endif
#endif

#ifndef ChineseIslPath
  #ifdef CompilerPath
    #if FileExists(CompilerPath + "\Languages\ChineseSimplified.isl")
      #define ChineseIslPath CompilerPath + "\Languages\ChineseSimplified.isl"
    #endif
  #endif
#endif

; /D 显式传入的路径也复核一遍：传错了同样降级为英文，而不是中断编译
#ifdef ChineseIslPath
  #if FileExists(ChineseIslPath)
    #define HaveChineseIsl
  #endif
#endif

[Setup]
; AppId 一经发布不可更改，改了会被当成另一个产品，无法正确升级覆盖
AppId={{8F3B2C41-7D6A-4E19-9C55-2A0E5B7F14D3}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
VersionInfoVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\GutOralAxis Desktop
DefaultGroupName={#AppName}
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\{#AppExeName}

; 默认当前用户安装（不提权），用户可在首页改为全机安装
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; 仅 64 位
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; 体积优化：发布目录里有大量 .NET / PyTorch 二进制，实体压缩收益明显
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4

OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
DisableProgramGroupPage=yes
DisableReadyPage=no
ShowLanguageDialog=auto
WizardStyle=modern
; 安装前必须让用户看到研究用途声明
InfoBeforeFile=installer-notice.txt

; 安装器图标直接复用应用本体那一份，不在 packaging 下再存一份副本——两份迟早走样。
; 路径相对本 .iss 所在目录；该文件由 desktop/packaging/make_icon.py 生成。
SetupIconFile=..\src\GutOralAxis.Desktop\Assets\AppIcon.ico

[Languages]
; 中文排在前面，安装程序按系统区域自动选中文；缺 .isl 时整段跳过，只剩英文
#ifdef HaveChineseIsl
Name: "chinesesimplified"; MessagesFile: "{#ChineseIslPath}"
#endif
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
; 语言前缀必须对应 [Languages] 里真实存在的条目，否则 ISCC 报 Unknown language name
#ifdef HaveChineseIsl
chinesesimplified.CreateDesktopIcon=创建桌面快捷方式
chinesesimplified.LaunchApp=立即运行 {#AppNameZh}
chinesesimplified.WebView2Downloading=正在下载 Microsoft Edge WebView2 运行时…
chinesesimplified.WebView2Required=本程序的界面依赖 Microsoft Edge WebView2 运行时，当前系统未安装。安装程序将自动下载并安装它，此步骤需要联网。
chinesesimplified.WebView2Failed=WebView2 运行时下载失败。请先手动安装 WebView2 运行时后重新运行本安装程序，或联系提供方获取离线安装包。
chinesesimplified.KeepData=保留本机已生成的分析记录与数据库
#endif
english.CreateDesktopIcon=Create a desktop shortcut
english.LaunchApp=Launch {#AppName}
english.WebView2Downloading=Downloading the Microsoft Edge WebView2 runtime...
english.WebView2Required=This application requires the Microsoft Edge WebView2 runtime, which is not installed. Setup will download and install it (internet connection required).
english.WebView2Failed=Failed to download the WebView2 runtime. Install it manually and run setup again.
english.KeepData=Keep analysis reports and the local database

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; 整个发布目录原样打包。不要只挑 exe——自包含运行时、WebUI 资源、
; Runtime\Engine 三者缺一不可。
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppNameZh}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppNameZh}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppNameZh}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
; 先装 WebView2（仅在缺失且已成功下载时），再按需启动应用
Filename: "{tmp}\MicrosoftEdgeWebview2Setup.exe"; Parameters: "/silent /install"; \
  StatusMsg: "{cm:WebView2Downloading}"; Flags: waituntilterminated; Check: NeedsWebView2Install
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchApp}"; \
  Flags: nowait postinstall skipifsilent

[Code]
const
  { WebView2 运行时在 EdgeUpdate 下的固定客户端 GUID }
  WebView2ClientKey = 'Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';

var
  DownloadPage: TDownloadWizardPage;
  WebView2Missing: Boolean;
  WebView2Downloaded: Boolean;
  RemoveDataCheckBox: TNewCheckBox;

{ 三个位置任一存在即认为运行时可用：
  64 位机器上的 HKLM WOW6432Node、HKLM 原生视图、以及当前用户安装 }
function WebView2Installed: Boolean;
var
  Version: String;
begin
  Result := False;

  if RegQueryStringValue(HKEY_LOCAL_MACHINE,
       'SOFTWARE\WOW6432Node\' + WebView2ClientKey, 'pv', Version) then
    if (Version <> '') and (Version <> '0.0.0.0') then
      Result := True;

  if (not Result) and RegQueryStringValue(HKEY_LOCAL_MACHINE,
       'SOFTWARE\' + WebView2ClientKey, 'pv', Version) then
    if (Version <> '') and (Version <> '0.0.0.0') then
      Result := True;

  if (not Result) and RegQueryStringValue(HKEY_CURRENT_USER,
       'SOFTWARE\' + WebView2ClientKey, 'pv', Version) then
    if (Version <> '') and (Version <> '0.0.0.0') then
      Result := True;
end;

function NeedsWebView2Install: Boolean;
begin
  Result := WebView2Missing and WebView2Downloaded;
end;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  Result := True;
end;

procedure InitializeWizard;
begin
  WebView2Missing := not WebView2Installed;
  WebView2Downloaded := False;
  DownloadPage := CreateDownloadPage(
    SetupMessage(msgWizardPreparing), ExpandConstant('{cm:WebView2Downloading}'), @OnDownloadProgress);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpReady) and WebView2Missing then
  begin
    if MsgBox(ExpandConstant('{cm:WebView2Required}'), mbConfirmation, MB_OKCANCEL) = IDCANCEL then
    begin
      Result := False;
      Exit;
    end;

    DownloadPage.Clear;
    DownloadPage.Add('{#WebView2BootstrapUrl}', 'MicrosoftEdgeWebview2Setup.exe', '');
    DownloadPage.Show;
    try
      try
        DownloadPage.Download;
        WebView2Downloaded := True;
      except
        { 下载失败不阻断安装：应用本体仍可装上，用户手动补运行时即可 }
        SuppressibleMsgBox(ExpandConstant('{cm:WebView2Failed}'), mbError, MB_OK, IDOK);
        WebView2Downloaded := False;
      end;
    finally
      DownloadPage.Hide;
    end;
  end;
end;

{ 卸载时默认保留用户数据，只有显式勾选才删除 }
procedure InitializeUninstallProgressForm;
begin
  RemoveDataCheckBox := TNewCheckBox.Create(UninstallProgressForm);
  RemoveDataCheckBox.Parent := UninstallProgressForm.InnerPage;
  RemoveDataCheckBox.Left := UninstallProgressForm.StatusLabel.Left;
  RemoveDataCheckBox.Top := UninstallProgressForm.StatusLabel.Top +
    UninstallProgressForm.StatusLabel.Height + ScaleY(24);
  RemoveDataCheckBox.Width := UninstallProgressForm.InnerPage.ClientWidth - ScaleX(32);
  RemoveDataCheckBox.Caption := ExpandConstant('{cm:KeepData}');
  RemoveDataCheckBox.Checked := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    if (RemoveDataCheckBox <> nil) and (not RemoveDataCheckBox.Checked) then
    begin
      DataDir := ExpandConstant('{localappdata}\GutOralAxis');
      if DirExists(DataDir) then
        DelTree(DataDir, True, True, True);
    end;
  end;
end;
