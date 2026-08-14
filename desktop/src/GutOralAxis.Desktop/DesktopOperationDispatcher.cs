using System.Text.Json;
using GutOralAxis.Core.Api;
using GutOralAxis.Core.Devices;
using GutOralAxis.Core.Logging;
using GutOralAxis.Core.Messaging;
using GutOralAxis.Infrastructure.Database;
using GutOralAxis.Infrastructure.Engine;
using GutOralAxis.Infrastructure.Reports;

namespace GutOralAxis.Desktop;

public sealed record OpenedJsonFile(string FileName, string Content);

public sealed record SavedHostFile(string FileName);

public interface IWindowSystemServices
{
    Task<OpenedJsonFile?> OpenJsonAsync(CancellationToken cancellationToken);

    Task<SavedHostFile?> SaveJsonAsync(
        string suggestedName,
        string content,
        CancellationToken cancellationToken);

    Task<SavedHostFile?> ExportPdfAsync(CancellationToken cancellationToken);

    Task PrintAsync(CancellationToken cancellationToken);
}

public sealed class DesktopOperationDispatcher(
    PythonEngineManager engine,
    ReportStore reportStore,
    AuditRepository auditRepository,
    IDeviceAdapter deviceAdapter,
    IWindowSystemServices windowServices,
    string versionManifestPath,
    IAppLogger logger) : IBridgeOperationHandler
{
    public async Task<OperationResult> HandleAsync(
        BridgeRequest request,
        CancellationToken cancellationToken)
    {
        OperationResult result;
        try
        {
            result = ApiOperationCatalog.TryGet(request.Operation, out _)
                ? await engine.HandleAsync(request, cancellationToken)
                : request.Operation switch
            {
                "file.openJson" => await OpenJsonAsync(cancellationToken),
                "file.saveJson" => await SaveJsonAsync(request, cancellationToken),
                "report.save" => await SaveReportAsync(request, cancellationToken),
                "report.list" => await ListReportsAsync(cancellationToken),
                "report.exportPdf" => await ExportPdfAsync(cancellationToken),
                "report.print" => await PrintAsync(cancellationToken),
                "app.getVersion" => await GetVersionAsync(cancellationToken),
                "device.discover" => await DiscoverDevicesAsync(cancellationToken),
                _ => Error(403, "OPERATION_NOT_ALLOWED", "该桌面操作未被允许。", request.RequestId),
            };
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            await RecordAuditAsync(request.Operation, 499, CancellationToken.None);
            throw;
        }
        catch (Exception exception)
        {
            logger.Error("host.operation_failed", $"Host operation failed. operation={request.Operation}", exception);
            result = Error(500, "HOST_OPERATION_FAILED", "Windows 系统操作失败。", request.RequestId);
        }
        await RecordAuditAsync(request.Operation, result.Status, cancellationToken);
        return result;
    }

    private async Task RecordAuditAsync(
        string operation,
        int status,
        CancellationToken cancellationToken)
    {
        try
        {
            await auditRepository.RecordAsync(
                "desktop.bridge_operation",
                status is >= 200 and < 400 ? "success" : status == 499 ? "cancelled" : "error",
                entityType: "operation",
                entityId: operation,
                safeDetails: JsonSerializer.SerializeToElement(new { operation, status }),
                cancellationToken: cancellationToken);
        }
        catch (Exception exception)
        {
            logger.Error("audit.write_failed", "Desktop operation audit could not be recorded.", exception);
        }
    }

    private async Task<OperationResult> OpenJsonAsync(CancellationToken cancellationToken)
    {
        var file = await windowServices.OpenJsonAsync(cancellationToken);
        return file is null
            ? Success(new Dictionary<string, object?> { ["cancelled"] = true })
            : Success(new Dictionary<string, object?>
            {
                ["cancelled"] = false,
                ["file_name"] = file.FileName,
                ["content"] = file.Content,
            });
    }

    private async Task<OperationResult> SaveJsonAsync(
        BridgeRequest request,
        CancellationToken cancellationToken)
    {
        var payload = RequirePayloadObject(request);
        var suggestedName = OptionalString(payload, "suggested_name") ?? "gut-oral-axis-data.json";
        if (!payload.TryGetProperty("content", out var content))
        {
            return Error(400, "INVALID_INPUT", "缺少要保存的 JSON 内容。", request.RequestId);
        }
        var text = content.ValueKind == JsonValueKind.String
            ? content.GetString() ?? string.Empty
            : JsonSerializer.Serialize(content, new JsonSerializerOptions(BridgeJson.Options) { WriteIndented = true });
        var saved = await windowServices.SaveJsonAsync(suggestedName, text + Environment.NewLine, cancellationToken);
        return saved is null
            ? Success(new Dictionary<string, object?> { ["cancelled"] = true })
            : Success(new Dictionary<string, object?>
            {
                ["cancelled"] = false,
                ["file_name"] = saved.FileName,
            });
    }

    private async Task<OperationResult> SaveReportAsync(
        BridgeRequest request,
        CancellationToken cancellationToken)
    {
        var payload = RequirePayloadObject(request);
        if (!payload.TryGetProperty("report", out var report))
        {
            return Error(400, "INVALID_INPUT", "缺少结构化报告内容。", request.RequestId);
        }
        var name = OptionalString(payload, "suggested_name") ?? "gut-oral-axis-report.json";
        var stored = await reportStore.SaveJsonAsync(report, name, cancellationToken: cancellationToken);
        return Success(new Dictionary<string, object?>
        {
            ["report_id"] = stored.Id,
            ["display_name"] = stored.DisplayName,
            ["display_location"] = $"本地报告中心 / {stored.DisplayName}",
            ["sha256"] = stored.Sha256,
        });
    }

    private async Task<OperationResult> ListReportsAsync(CancellationToken cancellationToken)
    {
        var reports = await reportStore.ListAsync(cancellationToken: cancellationToken);
        return Success(new Dictionary<string, object?>
        {
            ["reports"] = reports.Select(report => new
            {
                report_id = report.Id,
                display_name = report.DisplayName,
                sha256 = report.Sha256,
                created_utc = report.CreatedUtc,
            }).ToArray(),
        });
    }

    private async Task<OperationResult> ExportPdfAsync(CancellationToken cancellationToken)
    {
        var saved = await windowServices.ExportPdfAsync(cancellationToken);
        return Success(new Dictionary<string, object?>
        {
            ["cancelled"] = saved is null,
            ["exported"] = saved is not null,
            ["file_name"] = saved?.FileName,
        });
    }

    private async Task<OperationResult> PrintAsync(CancellationToken cancellationToken)
    {
        await windowServices.PrintAsync(cancellationToken);
        return Success(new Dictionary<string, object?> { ["print_dialog_opened"] = true });
    }

    private async Task<OperationResult> GetVersionAsync(CancellationToken cancellationToken)
    {
        var text = await File.ReadAllTextAsync(versionManifestPath, cancellationToken);
        using var document = JsonDocument.Parse(text);
        return Success(new Dictionary<string, object?>
        {
            ["versions"] = document.RootElement.Clone(),
        });
    }

    private async Task<OperationResult> DiscoverDevicesAsync(CancellationToken cancellationToken)
    {
        var devices = await deviceAdapter.DiscoverAsync(cancellationToken);
        return Success(new Dictionary<string, object?>
        {
            ["adapter"] = deviceAdapter.AdapterName,
            ["devices"] = devices,
            ["protocol_configured"] = devices.Count > 0,
        });
    }

    private static JsonElement RequirePayloadObject(BridgeRequest request)
    {
        if (request.Payload is not { ValueKind: JsonValueKind.Object } payload)
        {
            throw new InvalidDataException("Host operation payload must be a JSON object.");
        }
        return payload;
    }

    private static string? OptionalString(JsonElement payload, string name) =>
        payload.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;

    private static OperationResult Success(IReadOnlyDictionary<string, object?> fields)
    {
        var payload = new Dictionary<string, object?>(fields, StringComparer.Ordinal)
        {
            ["status"] = "success",
        };
        return new OperationResult(200, JsonSerializer.SerializeToElement(payload, BridgeJson.Options));
    }

    private static OperationResult Error(
        int status,
        string code,
        string message,
        string requestId) =>
        new(
            status,
            JsonSerializer.SerializeToElement(new
            {
                status = "error",
                error_code = code,
                message,
                request_id = requestId,
                details = Array.Empty<object>(),
            }, BridgeJson.Options));
}
