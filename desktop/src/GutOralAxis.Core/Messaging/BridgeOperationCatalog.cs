using GutOralAxis.Core.Api;

namespace GutOralAxis.Core.Messaging;

public static class BridgeOperationCatalog
{
    private static readonly IReadOnlySet<string> HostOperations = new HashSet<string>(
        new[]
        {
            "file.openJson",
            "file.saveJson",
            "report.save",
            "report.list",
            "report.exportPdf",
            "report.print",
            "app.getVersion",
            "device.discover",
        },
        StringComparer.Ordinal);

    public static bool IsAllowed(string operation) =>
        ApiOperationCatalog.TryGet(operation, out _) || HostOperations.Contains(operation);

    public static bool IsHostOperation(string operation) => HostOperations.Contains(operation);
}
