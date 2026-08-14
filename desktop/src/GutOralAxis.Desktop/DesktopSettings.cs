using System.Text.Json;
using System.Text.Json.Serialization;

namespace GutOralAxis.Desktop;

public sealed record DesktopSettings(
    [property: JsonPropertyName("enable_internal_oral_adenoma")]
    bool EnableInternalOralAdenoma,
    [property: JsonPropertyName("allow_development_engine_fallback")]
    bool AllowDevelopmentEngineFallback)
{
    public static DesktopSettings Load(string path)
    {
        if (!File.Exists(path))
        {
            return new DesktopSettings(false, false);
        }
        var settings = JsonSerializer.Deserialize<DesktopSettings>(
            File.ReadAllText(path),
            new JsonSerializerOptions(JsonSerializerDefaults.Web));
        return settings ?? throw new InvalidDataException("Desktop settings are empty.");
    }
}
