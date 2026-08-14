namespace GutOralAxis.Infrastructure;

public sealed record AppPaths(
    string Root,
    string Data,
    string Database,
    string Reports,
    string Logs,
    string Runtime,
    string WebView2)
{
    public static AppPaths Create(string? rootOverride = null)
    {
        if (string.IsNullOrWhiteSpace(rootOverride))
        {
            rootOverride = Environment.GetEnvironmentVariable("GOA_DESKTOP_DATA_ROOT");
        }
        var root = Path.GetFullPath(
            rootOverride
            ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "GutOralAxis",
                "Desktop"));
        var paths = new AppPaths(
            root,
            Path.Combine(root, "Data"),
            Path.Combine(root, "Data", "Database"),
            Path.Combine(root, "Data", "Reports"),
            Path.Combine(root, "Logs"),
            Path.Combine(root, "Runtime"),
            Path.Combine(root, "WebView2"));

        foreach (var path in new[]
        {
            paths.Root,
            paths.Data,
            paths.Database,
            paths.Reports,
            paths.Logs,
            paths.Runtime,
            paths.WebView2,
        })
        {
            Directory.CreateDirectory(path);
        }

        return paths;
    }
}
