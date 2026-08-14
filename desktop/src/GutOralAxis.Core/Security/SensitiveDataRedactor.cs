using System.Text;
using System.Text.Json;

namespace GutOralAxis.Core.Security;

public sealed class SensitiveDataRedactor(IEnumerable<string>? additionalKeys = null)
{
    private const string Replacement = "[REDACTED]";
    private readonly HashSet<string> sensitiveKeys = new(
        new[]
        {
            "patient_id",
            "patientId",
            "name",
            "phone",
            "email",
            "address",
            "current_medications",
            "drug_allergies",
            "microbiome",
            "microbes",
            "oral_abundances",
            "payload",
        }.Concat(additionalKeys ?? Array.Empty<string>()),
        StringComparer.OrdinalIgnoreCase);

    public string RedactJson(string json)
    {
        try
        {
            using var document = JsonDocument.Parse(json, new JsonDocumentOptions { MaxDepth = 64 });
            using var stream = new MemoryStream();
            using (var writer = new Utf8JsonWriter(stream))
            {
                WriteElement(writer, document.RootElement, null);
            }
            return Encoding.UTF8.GetString(stream.ToArray());
        }
        catch (JsonException)
        {
            return Replacement;
        }
    }

    private void WriteElement(Utf8JsonWriter writer, JsonElement element, string? propertyName)
    {
        if (propertyName is not null && sensitiveKeys.Contains(propertyName))
        {
            writer.WriteStringValue(Replacement);
            return;
        }

        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                writer.WriteStartObject();
                foreach (var property in element.EnumerateObject())
                {
                    writer.WritePropertyName(property.Name);
                    WriteElement(writer, property.Value, property.Name);
                }
                writer.WriteEndObject();
                break;
            case JsonValueKind.Array:
                writer.WriteStartArray();
                foreach (var item in element.EnumerateArray())
                {
                    WriteElement(writer, item, propertyName);
                }
                writer.WriteEndArray();
                break;
            default:
                element.WriteTo(writer);
                break;
        }
    }
}
