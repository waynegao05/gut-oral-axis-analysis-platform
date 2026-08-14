using System.Text.Json;
using System.Text.Json.Serialization;

namespace GutOralAxis.Core.Messaging;

public sealed record BridgeRequest(
    string RequestId,
    string Operation,
    JsonElement? Payload);

public sealed record OperationResult(int Status, JsonElement Payload);

public sealed record BridgeResponse(
    [property: JsonPropertyName("type")] string Type,
    [property: JsonPropertyName("version")] int Version,
    [property: JsonPropertyName("requestId")] string RequestId,
    [property: JsonPropertyName("status")] int Status,
    [property: JsonPropertyName("payload")] JsonElement Payload)
{
    public const string MessageType = "goa.response";
    public const int ProtocolVersion = 1;

    public static BridgeResponse FromResult(string requestId, OperationResult result) =>
        new(MessageType, ProtocolVersion, requestId, result.Status, result.Payload);

    public static BridgeResponse Error(
        string requestId,
        int status,
        string errorCode,
        string message) =>
        new(
            MessageType,
            ProtocolVersion,
            requestId,
            status,
            JsonSerializer.SerializeToElement(new
            {
                status = "error",
                error_code = errorCode,
                message,
                request_id = requestId,
                details = Array.Empty<object>(),
            }));

    public string ToJson() => JsonSerializer.Serialize(this, BridgeJson.Options);
}

public static class BridgeJson
{
    public static JsonSerializerOptions Options { get; } = new(JsonSerializerDefaults.Web)
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        MaxDepth = 64,
    };
}
