namespace GutOralAxis.Infrastructure.Engine;

public sealed record PythonEngineOptions(
    string PythonExecutable,
    string WorkingDirectory,
    TimeSpan StartupTimeout,
    TimeSpan RequestTimeout,
    int MaximumResponseBytes = 8 * 1024 * 1024,
    bool IncludeEngineDiagnosticText = false,
    IReadOnlyList<string>? BootstrapArguments = null,
    IReadOnlyDictionary<string, string>? EnvironmentVariables = null)
{
    public static PythonEngineOptions CreateDevelopment(
        string applicationBaseDirectory,
        string repositoryRoot,
        bool allowDevelopmentFallback = true)
    {
        var bundledEngine = Path.Combine(
            applicationBaseDirectory,
            "Runtime",
            "Engine",
            "goa-ai-engine.exe");
        if (File.Exists(bundledEngine))
        {
            return new PythonEngineOptions(
                bundledEngine,
                Path.GetDirectoryName(bundledEngine)!,
                TimeSpan.FromMinutes(3),
                TimeSpan.FromMinutes(2),
                BootstrapArguments: Array.Empty<string>());
        }
        if (!allowDevelopmentFallback)
        {
            throw new FileNotFoundException(
                "The packaged AI Engine is required but was not found.",
                bundledEngine);
        }

        var bundledPython = Path.Combine(
            applicationBaseDirectory,
            "runtime",
            "python",
            "python.exe");
        var executable = Environment.GetEnvironmentVariable("GOA_DESKTOP_PYTHON");
        if (string.IsNullOrWhiteSpace(executable))
        {
            executable = File.Exists(bundledPython) ? bundledPython : "python";
        }

        var workingDirectory = Environment.GetEnvironmentVariable("GOA_DESKTOP_ENGINE_ROOT");
        if (string.IsNullOrWhiteSpace(workingDirectory))
        {
            workingDirectory = repositoryRoot;
        }

        return new PythonEngineOptions(
            executable,
            Path.GetFullPath(workingDirectory),
            TimeSpan.FromMinutes(3),
            TimeSpan.FromMinutes(2),
            IncludeEngineDiagnosticText: string.Equals(
                Environment.GetEnvironmentVariable("GOA_DESKTOP_VERBOSE_ENGINE_LOGS"),
                "1",
                StringComparison.Ordinal),
            BootstrapArguments: null);
    }
}
