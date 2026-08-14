using System.Text.Json;
using GutOralAxis.Core.Api;
using GutOralAxis.Core.Messaging;
using GutOralAxis.Core.Security;

var tests = new (string Name, Func<Task> Run)[]
{
    ("allowed bridge request", TestAllowedBridgeRequest),
    ("unknown operation rejected", TestUnknownOperation),
    ("host operation allowed", TestHostOperation),
    ("oversized message rejected", TestOversizedMessage),
    ("router correlation", TestRouterCorrelation),
    ("path traversal rejected", TestPathTraversal),
    ("sensitive JSON redacted", TestRedaction),
    ("WebView2 origin is exact", TestWebViewOrigin),
};

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
        failures.Add($"FAIL {test.Name}: {exception.Message}");
    }
}

foreach (var failure in failures)
{
    Console.Error.WriteLine(failure);
}

Console.WriteLine($"{tests.Length - failures.Count}/{tests.Length} core smoke tests passed.");
return failures.Count == 0 ? 0 : 1;

static Task TestAllowedBridgeRequest()
{
    const string message =
        "{\"type\":\"goa.request\",\"version\":1,\"requestId\":\"req-1\","
        + "\"operation\":\"analyze\",\"payload\":{\"age\":52}}";
    Assert(BridgeRequestParser.TryParse(message, out var request, out _), "Request should parse.");
    Assert(request!.RequestId == "req-1", "Request ID changed.");
    Assert(request.Operation == "analyze", "Operation changed.");
    Assert(ApiOperationCatalog.TryGet(request.Operation, out var operation), "Operation missing.");
    Assert(operation.EnginePath == "/api/v1/analyze", "Engine path changed.");
    Assert(ApiOperationCatalog.TryGet("predict", out var predict), "Predict operation missing.");
    Assert(predict.EnginePath == "/api/v1/predict", "Predict Engine path changed.");
    return Task.CompletedTask;
}

static Task TestUnknownOperation()
{
    const string message =
        "{\"type\":\"goa.request\",\"version\":1,\"requestId\":\"req-2\","
        + "\"operation\":\"shell.execute\"}";
    Assert(!BridgeRequestParser.TryParse(message, out _, out var error), "Unknown operation passed.");
    Assert(error!.Status == 403, "Unknown operation did not return 403.");
    Assert(error.Payload.GetProperty("error_code").GetString() == "OPERATION_NOT_ALLOWED", "Wrong code.");
    return Task.CompletedTask;
}

static Task TestHostOperation()
{
    const string message =
        "{\"type\":\"goa.request\",\"version\":1,\"requestId\":\"req-host\","
        + "\"operation\":\"file.openJson\"}";
    Assert(BridgeRequestParser.TryParse(message, out var request, out _), "Host operation was rejected.");
    Assert(request!.Operation == "file.openJson", "Host operation changed.");
    return Task.CompletedTask;
}

static Task TestOversizedMessage()
{
    var message = new string('x', BridgeRequestParser.MaxMessageBytes + 1);
    Assert(!BridgeRequestParser.TryParse(message, out _, out var error), "Oversized request passed.");
    Assert(error!.Status == 413, "Oversized request did not return 413.");
    return Task.CompletedTask;
}

static async Task TestRouterCorrelation()
{
    var handler = new EchoHandler();
    var router = new BridgeRouter(handler);
    const string message =
        "{\"type\":\"goa.request\",\"version\":1,\"requestId\":\"req-3\","
        + "\"operation\":\"standardize\",\"payload\":{\"value\":1}}";
    var responseJson = await router.RouteAsync(message, CancellationToken.None);
    using var response = JsonDocument.Parse(responseJson);
    Assert(response.RootElement.GetProperty("requestId").GetString() == "req-3", "Correlation lost.");
    Assert(response.RootElement.GetProperty("status").GetInt32() == 200, "Wrong status.");
}

static Task TestPathTraversal()
{
    var root = Path.Combine(Path.GetTempPath(), "goa-path-root");
    AssertThrows<InvalidOperationException>(() => PathPolicy.ResolveWithin(root, "../secret.json"));
    var child = PathPolicy.ResolveWithin(root, "reports/report.json");
    Assert(child.StartsWith(Path.GetFullPath(root), StringComparison.OrdinalIgnoreCase), "Child escaped root.");
    return Task.CompletedTask;
}

static Task TestRedaction()
{
    var redactor = new SensitiveDataRedactor();
    var redacted = redactor.RedactJson(
        "{\"patient_id\":\"P1\",\"age\":52,\"metadata\":{\"drug_allergies\":[\"x\"]}}" );
    Assert(!redacted.Contains("P1", StringComparison.Ordinal), "Patient ID leaked.");
    Assert(!redacted.Contains("\\\"x\\\"", StringComparison.Ordinal), "Allergy leaked.");
    Assert(redacted.Contains("52", StringComparison.Ordinal), "Non-sensitive field was removed.");
    return Task.CompletedTask;
}

static Task TestWebViewOrigin()
{
    const string host = "app.gutoralaxis.local";
    Assert(
        WebViewOriginPolicy.IsAllowedApplicationSource(
            "https://app.gutoralaxis.local/index.html",
            host),
        "Application origin was rejected.");
    foreach (var source in new[]
    {
        "http://app.gutoralaxis.local/index.html",
        "https://app.gutoralaxis.local:444/index.html",
        "https://app.gutoralaxis.local.evil.example/index.html",
        "https://user@app.gutoralaxis.local/index.html",
        "not-a-uri",
    })
    {
        Assert(
            !WebViewOriginPolicy.IsAllowedApplicationSource(source, host),
            $"Untrusted WebView2 source was accepted: {source}");
    }
    return Task.CompletedTask;
}

static void Assert(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
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

file sealed class EchoHandler : IBridgeOperationHandler
{
    public Task<OperationResult> HandleAsync(
        BridgeRequest request,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var payload = JsonSerializer.SerializeToElement(new
        {
            status = "success",
            operation = request.Operation,
        });
        return Task.FromResult(new OperationResult(200, payload));
    }
}
