using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using GutOralAxis.Core.Api;
using GutOralAxis.Core.Logging;
using GutOralAxis.Core.Messaging;

namespace GutOralAxis.Infrastructure.Engine;

public sealed class PythonEngineManager : IBridgeOperationHandler, IAsyncDisposable
{
    private const string LoopbackHost = "127.0.0.1";
    private readonly PythonEngineOptions options;
    private readonly IAppLogger logger;
    private readonly SemaphoreSlim lifecycleGate = new(1, 1);
    private Process? process;
    private HttpClient? client;
    private string? token;
    private int port;
    private bool disposed;

    public PythonEngineManager(PythonEngineOptions options, IAppLogger logger)
    {
        this.options = options;
        this.logger = logger;
        if (!Directory.Exists(options.WorkingDirectory))
        {
            throw new DirectoryNotFoundException(options.WorkingDirectory);
        }
        if (options.StartupTimeout <= TimeSpan.Zero || options.RequestTimeout <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Engine timeouts must be positive.");
        }
    }

    public bool IsRunning => process is { HasExited: false } && client is not null;

    public async Task StartAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        await lifecycleGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (IsRunning)
            {
                return;
            }

            await StopOwnedProcessAsync().ConfigureAwait(false);
            port = ReserveLoopbackPort();
            token = Convert.ToHexString(RandomNumberGenerator.GetBytes(32));
            var startInfo = new ProcessStartInfo
            {
                FileName = options.PythonExecutable,
                WorkingDirectory = options.WorkingDirectory,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            var bootstrapArguments = options.BootstrapArguments ?? new[] { "-m", "ai_engine" };
            foreach (var argument in bootstrapArguments)
            {
                startInfo.ArgumentList.Add(argument);
            }
            startInfo.ArgumentList.Add("--host");
            startInfo.ArgumentList.Add(LoopbackHost);
            startInfo.ArgumentList.Add("--port");
            startInfo.ArgumentList.Add(port.ToString(System.Globalization.CultureInfo.InvariantCulture));
            startInfo.Environment["GOA_ENGINE_TOKEN"] = token;
            startInfo.Environment["PYTHONUTF8"] = "1";
            startInfo.Environment["PYTHONUNBUFFERED"] = "1";
            foreach (var (name, value) in options.EnvironmentVariables
                ?? new Dictionary<string, string>())
            {
                if (name is "GOA_ENGINE_TOKEN" or "GOA_ENGINE_HOST" or "GOA_ENGINE_PORT")
                {
                    throw new InvalidOperationException($"Reserved Engine environment variable: {name}");
                }
                startInfo.Environment[name] = value;
            }

            process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
            process.OutputDataReceived += (_, eventArgs) => LogEngineLine("engine.stdout", eventArgs.Data);
            process.ErrorDataReceived += (_, eventArgs) => LogEngineLine("engine.stderr", eventArgs.Data);
            process.Exited += (_, _) => logger.Warning("engine.exited", "Python Engine process exited.");
            if (!process.Start())
            {
                throw new InvalidOperationException("Unable to start Python Engine.");
            }
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            var handler = new SocketsHttpHandler
            {
                UseProxy = false,
                ConnectTimeout = TimeSpan.FromSeconds(5),
            };
            client = new HttpClient(handler)
            {
                BaseAddress = new Uri($"http://{LoopbackHost}:{port}"),
                Timeout = Timeout.InfiniteTimeSpan,
            };
            client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
            client.DefaultRequestHeaders.Add("X-GOA-Engine-Token", token);

            using var startupTimeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            startupTimeout.CancelAfter(options.StartupTimeout);
            await WaitForHealthAsync(startupTimeout.Token).ConfigureAwait(false);
            logger.Information("engine.started", "Python Engine is healthy on a loopback port.");
        }
        catch
        {
            await StopOwnedProcessAsync().ConfigureAwait(false);
            throw;
        }
        finally
        {
            lifecycleGate.Release();
        }
    }

    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        await lifecycleGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await StopOwnedProcessAsync().ConfigureAwait(false);
        }
        finally
        {
            lifecycleGate.Release();
        }
    }

    public async Task<OperationResult> HandleAsync(
        BridgeRequest request,
        CancellationToken cancellationToken)
    {
        if (!ApiOperationCatalog.TryGet(request.Operation, out var operation))
        {
            return ErrorResult(403, "OPERATION_NOT_ALLOWED", "该分析操作未被允许。", request.RequestId);
        }
        if (!IsRunning || client is null)
        {
            return ErrorResult(503, "PYTHON_ENGINE_OFFLINE", "本地分析引擎尚未就绪。", request.RequestId);
        }

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(options.RequestTimeout);
        using var message = new HttpRequestMessage(operation.Method, operation.EnginePath);
        if (operation.Method == HttpMethod.Post)
        {
            var json = request.Payload?.GetRawText() ?? "{}";
            message.Content = new StringContent(json, Encoding.UTF8, "application/json");
        }

        try
        {
            using var response = await client.SendAsync(
                message,
                HttpCompletionOption.ResponseHeadersRead,
                timeout.Token).ConfigureAwait(false);
            var bytes = await response.Content.ReadAsByteArrayAsync(timeout.Token).ConfigureAwait(false);
            if (bytes.Length > options.MaximumResponseBytes)
            {
                return ErrorResult(502, "ENGINE_RESPONSE_TOO_LARGE", "本地分析引擎返回的数据过大。", request.RequestId);
            }

            try
            {
                using var document = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 64 });
                return new OperationResult((int)response.StatusCode, document.RootElement.Clone());
            }
            catch (JsonException)
            {
                return ErrorResult(502, "INVALID_ENGINE_RESPONSE", "本地分析引擎返回了无法解析的数据。", request.RequestId);
            }
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return ErrorResult(504, "ENGINE_TIMEOUT", "本地分析超时，请稍后重试。", request.RequestId);
        }
        catch (HttpRequestException exception)
        {
            logger.Error("engine.request_failed", "Python Engine request failed.", exception);
            return ErrorResult(503, "PYTHON_ENGINE_OFFLINE", "本地分析引擎连接中断。", request.RequestId);
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (disposed)
        {
            return;
        }
        disposed = true;
        await StopAsync().ConfigureAwait(false);
        lifecycleGate.Dispose();
    }

    private async Task WaitForHealthAsync(CancellationToken cancellationToken)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (process is null || process.HasExited)
            {
                throw new InvalidOperationException("Python Engine exited before becoming healthy.");
            }

            try
            {
                using var response = await client!.GetAsync("/api/v1/health", cancellationToken).ConfigureAwait(false);
                if (response.IsSuccessStatusCode)
                {
                    using var document = JsonDocument.Parse(
                        await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false));
                    if (document.RootElement.TryGetProperty("engine_ready", out var ready) && ready.GetBoolean())
                    {
                        return;
                    }
                }
            }
            catch (HttpRequestException)
            {
            }
            await Task.Delay(TimeSpan.FromMilliseconds(250), cancellationToken).ConfigureAwait(false);
        }
    }

    private async Task StopOwnedProcessAsync()
    {
        client?.Dispose();
        client = null;
        token = null;
        var ownedProcess = process;
        process = null;
        if (ownedProcess is null)
        {
            return;
        }

        try
        {
            if (!ownedProcess.HasExited)
            {
                ownedProcess.Kill(entireProcessTree: true);
                await ownedProcess.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(10)).ConfigureAwait(false);
            }
        }
        catch (InvalidOperationException)
        {
        }
        catch (System.ComponentModel.Win32Exception exception)
        {
            logger.Error("engine.stop_failed", "Unable to stop the owned Python Engine process.", exception);
        }
        catch (TimeoutException)
        {
            logger.Warning("engine.stop_timeout", "Timed out while stopping the owned Python Engine process.");
        }
        finally
        {
            ownedProcess.Dispose();
        }
    }

    private void LogEngineLine(string eventName, string? line)
    {
        if (!string.IsNullOrWhiteSpace(line))
        {
            var message = options.IncludeEngineDiagnosticText
                ? line.Length <= 2_000 ? line : line[..2_000]
                : "Python Engine emitted a diagnostic line; text suppressed by default.";
            logger.Information(eventName, message);
        }
    }

    private static OperationResult ErrorResult(
        int statusCode,
        string errorCode,
        string message,
        string requestId) =>
        new(
            statusCode,
            JsonSerializer.SerializeToElement(new
            {
                status = "error",
                error_code = errorCode,
                message,
                request_id = requestId,
                details = Array.Empty<object>(),
            }));

    private static int ReserveLoopbackPort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        try
        {
            return ((IPEndPoint)listener.LocalEndpoint).Port;
        }
        finally
        {
            listener.Stop();
        }
    }
}
