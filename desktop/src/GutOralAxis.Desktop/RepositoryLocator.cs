namespace GutOralAxis.Desktop;

public static class RepositoryLocator
{
    public static string FindEngineRoot(string applicationBaseDirectory)
    {
        var bundled = Path.Combine(applicationBaseDirectory, "Runtime", "Engine");
        if (File.Exists(Path.Combine(bundled, "goa-ai-engine.exe"))
            || File.Exists(Path.Combine(bundled, "ai_engine", "__main__.py")))
        {
            return bundled;
        }

        var current = new DirectoryInfo(Path.GetFullPath(applicationBaseDirectory));
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "ai_engine", "__main__.py")))
            {
                return current.FullName;
            }
            current = current.Parent;
        }

        throw new DirectoryNotFoundException(
            "Unable to locate the packaged or development Python Engine root.");
    }
}
