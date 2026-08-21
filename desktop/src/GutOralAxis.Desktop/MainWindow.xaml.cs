using System.Diagnostics;
using GutOralAxis.Core.Logging;
using GutOralAxis.Core.Messaging;
using GutOralAxis.Core.Security;
using GutOralAxis.Infrastructure;
using GutOralAxis.Infrastructure.Database;
using GutOralAxis.Infrastructure.Engine;
using GutOralAxis.Infrastructure.Devices;
using GutOralAxis.Infrastructure.Logging;
using GutOralAxis.Infrastructure.Reports;
using Microsoft.UI.Xaml;
using Microsoft.Web.WebView2.Core;
using Windows.Storage.Pickers;
using WinRT.Interop;

namespace GutOralAxis.Desktop;

public sealed partial class MainWindow : Window, IWindowSystemServices
{
    private const string ApplicationHost = "app.gutoralaxis.local";
    private const string ApplicationOrigin = $"https://{ApplicationHost}/";
    private static readonly IReadOnlySet<string> AllowedExternalHosts = new HashSet<string>(
        new[]
        {
            "api.fda.gov",
            "dailymed.nlm.nih.gov",
            "doi.org",
            "gastro.org",
            "lhncbc.nlm.nih.gov",
            "open.fda.gov",
            "pmc.ncbi.nlm.nih.gov",
            "rxnav.nlm.nih.gov",
            "www.cdc.gov",
            "www.fda.gov",
            "www.uspreventiveservicestaskforce.org",
            "www.who.int",
        },
        StringComparer.OrdinalIgnoreCase);
    private readonly AppPaths paths;
    private readonly DesktopSettings settings;
    private readonly IAppLogger logger;
    private readonly CancellationTokenSource lifetimeCancellation = new();
    private PythonEngineManager? engine;
    private RollingFileLogger? ownedLogger;
    private BridgeRouter? bridgeRouter;
    private bool closing;

    public MainWindow(AppPaths paths, DesktopSettings settings, IAppLogger logger)
    {
        this.paths = paths;
        this.settings = settings;
        this.logger = logger;
        ownedLogger = logger as RollingFileLogger;
        InitializeComponent();
        ConfigureWindowChrome();
        Closed += OnClosed;
    }

    public async Task InitializeAsync()
    {
        StartupStatus.Text = "正在初始化本地数据库……";
        var database = new SqliteDatabase(
            Path.Combine(paths.Database, "gut-oral-axis.db"),
            logger);
        await database.InitializeAsync(lifetimeCancellation.Token);
        var reportStore = new ReportStore(paths.Reports, database, logger);
        var auditRepository = new AuditRepository(database);

        StartupStatus.Text = "正在加载 AI 模型，此过程首次启动可能较慢……";
        var repositoryRoot = RepositoryLocator.FindEngineRoot(AppContext.BaseDirectory);
        var options = PythonEngineOptions.CreateDevelopment(
            AppContext.BaseDirectory,
            repositoryRoot,
            settings.AllowDevelopmentEngineFallback);
        if (settings.EnableInternalOralAdenoma)
        {
            options = options with
            {
                EnvironmentVariables = new Dictionary<string, string>
                {
                    ["GOA_ENABLE_INTERNAL_ORAL_ADENOMA"] = "1",
                },
            };
        }
        engine = new PythonEngineManager(options, logger);
        try
        {
            await engine.StartAsync(lifetimeCancellation.Token);
        }
        catch (Exception exception)
        {
            logger.Error("engine.start_failed", "Python Engine could not start.", exception);
        }

        var dispatcher = new DesktopOperationDispatcher(
            engine,
            reportStore,
            auditRepository,
            new NoDeviceAdapter(),
            this,
            Path.Combine(AppContext.BaseDirectory, "version-manifest.json"),
            logger);
        bridgeRouter = new BridgeRouter(dispatcher);

        StartupStatus.Text = "正在加载本地界面……";
        await InitializeWebViewAsync();
        StartupOverlay.Visibility = Visibility.Collapsed;
        logger.Information("application.ready", "Desktop application is ready.");
        if (string.Equals(
            Environment.GetEnvironmentVariable("GOA_DESKTOP_SMOKE_EXIT"),
            "1",
            StringComparison.Ordinal))
        {
            await Task.Delay(TimeSpan.FromMilliseconds(500));
            Close();
        }
    }

    public void ShowFatalStartupError(string message, string logDirectory)
    {
        StartupProgress.IsActive = false;
        StartupStatus.Text = $"{message}\n日志目录：{logDirectory}";
        StartupOverlay.Visibility = Visibility.Visible;
    }

    public async Task<OpenedJsonFile?> OpenJsonAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var picker = new FileOpenPicker();
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(this));
        picker.FileTypeFilter.Add(".json");
        var file = await picker.PickSingleFileAsync();
        if (file is null)
        {
            return null;
        }
        var info = new FileInfo(file.Path);
        if (info.Length > BridgeRequestParser.MaxMessageBytes)
        {
            throw new InvalidDataException("JSON 文件超过 2 MB 限制。");
        }
        var content = await File.ReadAllTextAsync(file.Path, cancellationToken);
        return new OpenedJsonFile(file.Name, content);
    }

    public async Task<SavedHostFile?> SaveJsonAsync(
        string suggestedName,
        string content,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var picker = new FileSavePicker
        {
            SuggestedFileName = Path.GetFileNameWithoutExtension(suggestedName),
        };
        picker.FileTypeChoices.Add("JSON", new List<string> { ".json" });
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(this));
        var file = await picker.PickSaveFileAsync();
        if (file is null)
        {
            return null;
        }
        await File.WriteAllTextAsync(file.Path, content, cancellationToken);
        return new SavedHostFile(file.Name);
    }

    public async Task<SavedHostFile?> ExportPdfAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var picker = new FileSavePicker { SuggestedFileName = $"gut-oral-axis-{DateTime.Now:yyyyMMdd}" };
        picker.FileTypeChoices.Add("PDF", new List<string> { ".pdf" });
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(this));
        var file = await picker.PickSaveFileAsync();
        if (file is null)
        {
            return null;
        }
        var printSettings = WebView.CoreWebView2.Environment.CreatePrintSettings();
        var exported = await WebView.CoreWebView2.PrintToPdfAsync(file.Path, printSettings);
        if (!exported)
        {
            throw new IOException("WebView2 failed to export the current page as PDF.");
        }
        return new SavedHostFile(file.Name);
    }

    public Task PrintAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        WebView.CoreWebView2.ShowPrintUI(CoreWebView2PrintDialogKind.System);
        return Task.CompletedTask;
    }

    private async Task InitializeWebViewAsync()
    {
        var webRoot = Path.Combine(AppContext.BaseDirectory, "WebUI");
        var indexPath = Path.Combine(webRoot, "index.html");
        if (!File.Exists(indexPath))
        {
            throw new FileNotFoundException(
                "Desktop WebUI is missing. Run the desktop WebUI build first.",
                indexPath);
        }

        var environment = await CoreWebView2Environment.CreateWithOptionsAsync(
            browserExecutableFolder: null,
            userDataFolder: paths.WebView2,
            options: new CoreWebView2EnvironmentOptions());
        await WebView.EnsureCoreWebView2Async(environment);
        var core = WebView.CoreWebView2;
        core.SetVirtualHostNameToFolderMapping(
            ApplicationHost,
            webRoot,
            CoreWebView2HostResourceAccessKind.DenyCors);
        core.Settings.AreDevToolsEnabled = IsDevelopmentMode();
        core.Settings.AreDefaultContextMenusEnabled = false;
        core.Settings.AreBrowserAcceleratorKeysEnabled = false;
        core.Settings.IsStatusBarEnabled = false;
        core.Settings.IsZoomControlEnabled = false;
        core.Settings.IsGeneralAutofillEnabled = false;
        core.Settings.IsPasswordAutosaveEnabled = false;
        core.NavigationStarting += OnNavigationStarting;
        core.NewWindowRequested += OnNewWindowRequested;
        core.PermissionRequested += (_, eventArgs) => eventArgs.State = CoreWebView2PermissionState.Deny;
        core.DownloadStarting += (_, eventArgs) => eventArgs.Cancel = true;
        core.WebMessageReceived += OnWebMessageReceived;
        WebView.Source = new Uri(ApplicationOrigin + "index.html");
    }

    private void ConfigureWindowChrome()
    {
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);

        var titleBar = AppWindow.TitleBar;
        titleBar.ButtonBackgroundColor = Microsoft.UI.Colors.Transparent;
        titleBar.ButtonInactiveBackgroundColor = Microsoft.UI.Colors.Transparent;

        var iconPath = Path.Combine(AppContext.BaseDirectory, "Assets", "AppIcon.ico");
        if (File.Exists(iconPath))
        {
            AppWindow.SetIcon(iconPath);
        }
        else
        {
            logger.Warning("application.icon_missing", "Desktop application icon is missing.");
        }
    }

    private void OnNavigationStarting(
        CoreWebView2 sender,
        CoreWebView2NavigationStartingEventArgs eventArgs)
    {
        if (eventArgs.Uri.StartsWith(ApplicationOrigin, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }
        eventArgs.Cancel = true;
        OpenAllowedExternalUri(eventArgs.Uri);
    }

    private void OnNewWindowRequested(
        CoreWebView2 sender,
        CoreWebView2NewWindowRequestedEventArgs eventArgs)
    {
        eventArgs.Handled = true;
        OpenAllowedExternalUri(eventArgs.Uri);
    }

    private async void OnWebMessageReceived(
        CoreWebView2 sender,
        CoreWebView2WebMessageReceivedEventArgs eventArgs)
    {
        if (!WebViewOriginPolicy.IsAllowedApplicationSource(eventArgs.Source, ApplicationHost))
        {
            logger.Warning(
                "webview.message_blocked",
                "Blocked a WebView2 message from outside the application origin.");
            return;
        }
        if (bridgeRouter is null)
        {
            return;
        }
        try
        {
            var response = await bridgeRouter.RouteAsync(
                eventArgs.WebMessageAsJson,
                lifetimeCancellation.Token);
            sender.PostWebMessageAsJson(response);
        }
        catch (OperationCanceledException) when (lifetimeCancellation.IsCancellationRequested)
        {
        }
    }

    private void OnClosed(object sender, WindowEventArgs args)
    {
        if (closing)
        {
            return;
        }
        closing = true;
        logger.Information("application.stop_requested", "Desktop application shutdown was requested.");
        try
        {
            lifetimeCancellation.Cancel();
            if (engine is not null)
            {
                engine.DisposeAsync().AsTask().GetAwaiter().GetResult();
            }
            logger.Information("application.stop", "Desktop application stopped.");
        }
        catch (Exception exception)
        {
            logger.Error("application.stop_failed", "Desktop shutdown encountered an error.", exception);
        }
        finally
        {
            ownedLogger?.Dispose();
            lifetimeCancellation.Dispose();
            (Application.Current as App)?.ReleaseSingleInstance();
        }
    }

    private void OpenAllowedExternalUri(string value)
    {
        if (!Uri.TryCreate(value, UriKind.Absolute, out var uri)
            || uri.Scheme != Uri.UriSchemeHttps
            || !AllowedExternalHosts.Contains(uri.IdnHost))
        {
            logger.Warning("webview.navigation_blocked", "Blocked an external navigation outside the evidence allowlist.");
            return;
        }

        Process.Start(new ProcessStartInfo(uri.AbsoluteUri) { UseShellExecute = true });
    }

    private static bool IsDevelopmentMode() =>
        string.Equals(
            Environment.GetEnvironmentVariable("GOA_DESKTOP_DEVTOOLS"),
            "1",
            StringComparison.Ordinal);

}
