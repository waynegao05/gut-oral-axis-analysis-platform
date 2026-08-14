using System.Text.Json;
using GutOralAxis.Core.Devices;
using GutOralAxis.Core.Messaging;
using GutOralAxis.Infrastructure;
using GutOralAxis.Infrastructure.Devices;
using GutOralAxis.Infrastructure.Engine;
using GutOralAxis.Infrastructure.Logging;

var testRoot = Environment.GetEnvironmentVariable("GOA_TEST_ROOT");
if (string.IsNullOrWhiteSpace(testRoot))
{
    Console.Error.WriteLine("GOA_TEST_ROOT must point to a writable test directory.");
    return 2;
}

testRoot = Path.Combine(Path.GetFullPath(testRoot), Guid.NewGuid().ToString("N"));
var tests = new List<(string Name, Func<Task> Run)>
{
    ("AppData paths created", TestAppPaths),
    ("rolling log redaction", TestRollingLogRedaction),
    ("no-device adapter is explicit", TestNoDeviceAdapter),
    ("packaged Engine executable is preferred", TestPackagedEngineOptions),
    ("production Engine fallback is refused", TestProductionEngineFallback),
};
if (Environment.GetEnvironmentVariable("GOA_TEST_ENGINE") == "1")
{
    tests.Add(("real Python Engine lifecycle", TestPythonEngine));
}

var failures = new List<string>();
foreach (var test in tests)
{
    try
    {
        await test.Run();
        Console.WriteLine($"PASS {test.Name}");
    }
    catch (Exception exception)
    {
        failures.Add($"FAIL {test.Name}: {exception}");
    }
}

foreach (var failure in failures)
{
    Console.Error.WriteLine(failure);
}
Console.WriteLine($"{tests.Count - failures.Count}/{tests.Count} infrastructure smoke tests passed.");
return failures.Count == 0 ? 0 : 1;

Task TestAppPaths()
{
    var paths = AppPaths.Create(Path.Combine(testRoot, "appdata"));
    foreach (var path in new[] { paths.Data, paths.Database, paths.Reports, paths.Logs, paths.Runtime, paths.WebView2 })
    {
        Assert(Directory.Exists(path), $"Missing directory: {path}");
    }
    return Task.CompletedTask;
}

Task TestRollingLogRedaction()
{
    var logDirectory = Path.Combine(testRoot, "logs");
    using (var logger = new RollingFileLogger(logDirectory))
    {
        logger.Information("json", "{\"patient_id\":\"P-SECRET\",\"age\":52}");
        logger.Information("text", "patient_id=P-SECRET message=ok");
    }

    var content = string.Join(Environment.NewLine, Directory.EnumerateFiles(logDirectory).Select(File.ReadAllText));
    Assert(!content.Contains("P-SECRET", StringComparison.Ordinal), "Sensitive patient ID leaked to logs.");
    Assert(content.Contains("52", StringComparison.Ordinal), "Safe diagnostic value was removed.");
    return Task.CompletedTask;
}

async Task TestNoDeviceAdapter()
{
    await using IDeviceAdapter adapter = new NoDeviceAdapter();
    var devices = await adapter.DiscoverAsync(CancellationToken.None);
    Assert(devices.Count == 0, "No-device adapter invented hardware.");
    await AssertThrowsAsync<NotSupportedException>(
        () => adapter.ConnectAsync("unknown", CancellationToken.None));
}

async Task TestPackagedEngineOptions()
{
    var applicationRoot = Path.Combine(testRoot, "packaged-app");
    var engineRoot = Path.Combine(applicationRoot, "Runtime", "Engine");
    Directory.CreateDirectory(engineRoot);
    var executable = Path.Combine(engineRoot, "goa-ai-engine.exe");
    await File.WriteAllBytesAsync(executable, []);
    var options = PythonEngineOptions.CreateDevelopment(applicationRoot, testRoot);
    Assert(options.PythonExecutable == executable, "Packaged Engine executable was not selected.");
    Assert(options.WorkingDirectory == engineRoot, "Packaged Engine working directory is incorrect.");
    Assert(options.BootstrapArguments?.Count == 0, "Packaged Engine received Python module arguments.");
}

Task TestProductionEngineFallback()
{
    var applicationRoot = Path.Combine(testRoot, "production-without-engine");
    Directory.CreateDirectory(applicationRoot);
    AssertThrows<FileNotFoundException>(() =>
        PythonEngineOptions.CreateDevelopment(
            applicationRoot,
            testRoot,
            allowDevelopmentFallback: false));
    return Task.CompletedTask;
}

async Task TestPythonEngine()
{
    var packagedEngine = Environment.GetEnvironmentVariable("GOA_DESKTOP_ENGINE_EXECUTABLE");
    var python = Environment.GetEnvironmentVariable("GOA_DESKTOP_PYTHON");
    var repository = Environment.GetEnvironmentVariable("GOA_DESKTOP_ENGINE_ROOT");
    PythonEngineOptions options;
    if (!string.IsNullOrWhiteSpace(packagedEngine))
    {
        var executable = Path.GetFullPath(packagedEngine);
        if (!File.Exists(executable))
        {
            throw new FileNotFoundException("Packaged Engine executable is missing.", executable);
        }
        options = new PythonEngineOptions(
            executable,
            Path.GetDirectoryName(executable)!,
            TimeSpan.FromMinutes(3),
            TimeSpan.FromMinutes(2),
            BootstrapArguments: Array.Empty<string>());
    }
    else if (!string.IsNullOrWhiteSpace(python) && !string.IsNullOrWhiteSpace(repository))
    {
        options = new PythonEngineOptions(
            python,
            repository,
            TimeSpan.FromMinutes(3),
            TimeSpan.FromMinutes(2));
    }
    else
    {
        throw new InvalidOperationException(
            "Engine smoke requires GOA_DESKTOP_ENGINE_EXECUTABLE or both GOA_DESKTOP_PYTHON and GOA_DESKTOP_ENGINE_ROOT.");
    }

    using var logger = new RollingFileLogger(Path.Combine(testRoot, "engine-logs"));
    await using var engine = new PythonEngineManager(options, logger);
    await engine.StartAsync();
    Assert(engine.IsRunning, "Engine did not reach running state.");

    var payload = JsonSerializer.SerializeToElement(new
    {
        microbes = new Dictionary<string, double>
        {
            ["Fusobacterium"] = 0.18,
            ["Porphyromonas"] = 0.15,
            ["Prevotella"] = 0.10,
            ["Streptococcus"] = 0.09,
            ["Lactobacillus"] = 0.02,
        },
        clinical = new
        {
            age = 52,
            sex = "Female",
            stage = 3,
            path_t = 3,
            path_n = 1,
            path_m = 0,
            tumor_location = "Colon Sigmoideum",
            tumor_morphology = "Adenocarcinoma",
        },
        metabolites = new
        {
            bile_acids = 0.8,
            scfa = 0.3,
            tryptophan_metabolism = 0.7,
        },
        metadata = new
        {
            current_medications = Array.Empty<string>(),
            drug_allergies = Array.Empty<string>(),
        },
    });
    var result = await engine.HandleAsync(
        new BridgeRequest("engine-smoke", "standardize", payload),
        CancellationToken.None);
    Assert(result.Status == 200, $"Engine returned {result.Status}: {result.Payload.GetRawText()}");
    Assert(result.Payload.GetProperty("status").GetString() == "success", "Unexpected engine envelope.");

    var bridge = new BridgeRouter(engine);
    var bridgeMessage = JsonSerializer.Serialize(new
    {
        type = "goa.request",
        version = 1,
        requestId = "engine-analysis-smoke",
        operation = "analyze",
        payload,
    }, BridgeJson.Options);
    var bridgeResponse = await bridge.RouteAsync(bridgeMessage, CancellationToken.None);
    using var responseDocument = JsonDocument.Parse(bridgeResponse);
    var responseRoot = responseDocument.RootElement;
    Assert(responseRoot.GetProperty("type").GetString() == "goa.response", "Unexpected bridge response type.");
    Assert(responseRoot.GetProperty("requestId").GetString() == "engine-analysis-smoke", "Bridge lost request correlation.");
    Assert(responseRoot.GetProperty("status").GetInt32() == 200, $"Analysis bridge failed: {bridgeResponse}");
    var responsePayload = responseRoot.GetProperty("payload");
    Assert(responsePayload.GetProperty("status").GetString() == "success", "Analysis did not return success.");
    Assert(responsePayload.TryGetProperty("report", out _), "Analysis response omitted the structured report.");
    await engine.StopAsync();
    Assert(!engine.IsRunning, "Engine process remained running after StopAsync.");
}

static void Assert(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static async Task AssertThrowsAsync<TException>(Func<Task> action)
    where TException : Exception
{
    try
    {
        await action();
    }
    catch (TException)
    {
        return;
    }
    throw new InvalidOperationException($"Expected {typeof(TException).Name}.");
}

static void AssertThrows<TException>(Action action)
    where TException : Exception
{
    try
    {
        action();
    }
    catch (TException)
    {
        return;
    }
    throw new InvalidOperationException($"Expected {typeof(TException).Name}.");
}
