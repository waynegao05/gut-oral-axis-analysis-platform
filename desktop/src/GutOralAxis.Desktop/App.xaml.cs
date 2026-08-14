using GutOralAxis.Infrastructure;
using GutOralAxis.Infrastructure.Logging;
using Microsoft.UI.Xaml;

namespace GutOralAxis.Desktop;

public partial class App : Application
{
    private readonly Mutex singleInstanceMutex;
    private readonly bool ownsMutex;
    private MainWindow? window;

    public App()
    {
        InitializeComponent();
        singleInstanceMutex = new Mutex(
            initiallyOwned: true,
            name: @"Local\GutOralAxis.Desktop",
            createdNew: out ownsMutex);
    }

    protected override async void OnLaunched(LaunchActivatedEventArgs args)
    {
        if (!ownsMutex)
        {
            Exit();
            return;
        }

        var paths = AppPaths.Create();
        var logger = new RollingFileLogger(paths.Logs);
        logger.Information("application.start", "Desktop application is starting.");
        try
        {
            var settings = DesktopSettings.Load(
                Path.Combine(AppContext.BaseDirectory, "desktop-settings.json"));
            window = new MainWindow(paths, settings, logger);
            window.Activate();
            await window.InitializeAsync();
        }
        catch (Exception exception)
        {
            logger.Error("application.initialization_failed", "Desktop initialization failed.", exception);
            if (window is not null)
            {
                window.ShowFatalStartupError("桌面程序初始化失败，请查看本地日志。", paths.Logs);
            }
            else
            {
                logger.Dispose();
                Exit();
            }
        }
    }

    public void ReleaseSingleInstance()
    {
        if (ownsMutex)
        {
            singleInstanceMutex.ReleaseMutex();
        }
        singleInstanceMutex.Dispose();
    }
}
