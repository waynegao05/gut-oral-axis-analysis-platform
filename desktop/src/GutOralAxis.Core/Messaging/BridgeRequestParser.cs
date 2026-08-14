using System.Text;
using System.Text.Json;

namespace GutOralAxis.Core.Messaging;

public static class BridgeRequestParser
{
    public const int MaxMessageBytes = 2 * 1024 * 1024;
    private const int MaxRequestIdLength = 128;

    public static bool TryParse(
        string json,
        out BridgeRequest? request,
        out BridgeResponse? error)
    {
        request = null;
        error = null;
        if (string.IsNullOrWhiteSpace(json))
        {
            error = BridgeResponse.Error("unknown", 400, "INVALID_MESSAGE", "桌面消息不能为空。");
            return false;
        }

        if (Encoding.UTF8.GetByteCount(json) > MaxMessageBytes)
        {
            error = BridgeResponse.Error("unknown", 413, "REQUEST_TOO_LARGE", "提交的数据超过桌面消息大小限制。");
            return false;
        }

        try
        {
            using var document = JsonDocument.Parse(
                json,
                new JsonDocumentOptions { MaxDepth = 64, CommentHandling = JsonCommentHandling.Disallow });
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                error = BridgeResponse.Error("unknown", 400, "INVALID_MESSAGE", "桌面消息必须是 JSON 对象。");
                return false;
            }

            var requestId = ReadString(root, "requestId");
            if (!IsValidRequestId(requestId))
            {
                error = BridgeResponse.Error("unknown", 400, "INVALID_REQUEST_ID", "桌面请求 ID 不合法。");
                return false;
            }

            if (ReadString(root, "type") != "goa.request" || ReadInt(root, "version") != 1)
            {
                error = BridgeResponse.Error(requestId!, 400, "UNSUPPORTED_PROTOCOL", "桌面消息协议版本不受支持。");
                return false;
            }

            var operation = ReadString(root, "operation");
            if (operation is null || !BridgeOperationCatalog.IsAllowed(operation))
            {
                error = BridgeResponse.Error(requestId!, 403, "OPERATION_NOT_ALLOWED", "该桌面操作未被允许。");
                return false;
            }

            JsonElement? payload = null;
            if (root.TryGetProperty("payload", out var payloadElement))
            {
                payload = payloadElement.Clone();
            }

            request = new BridgeRequest(requestId!, operation, payload);
            return true;
        }
        catch (JsonException)
        {
            error = BridgeResponse.Error("unknown", 400, "INVALID_JSON", "桌面消息不是有效 JSON。");
            return false;
        }
    }

    private static string? ReadString(JsonElement element, string name) =>
        element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;

    private static int? ReadInt(JsonElement element, string name) =>
        element.TryGetProperty(name, out var value) && value.TryGetInt32(out var result)
            ? result
            : null;

    private static bool IsValidRequestId(string? value)
    {
        if (string.IsNullOrWhiteSpace(value) || value.Length > MaxRequestIdLength)
        {
            return false;
        }

        return value.All(character =>
            char.IsAsciiLetterOrDigit(character) || character is '-' or '_' or '.');
    }
}
