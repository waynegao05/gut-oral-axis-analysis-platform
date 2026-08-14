namespace GutOralAxis.Core.Versioning;

public sealed record VersionManifest(
    string Application,
    string Frontend,
    string AiEngine,
    string Model,
    int DatabaseSchema);
