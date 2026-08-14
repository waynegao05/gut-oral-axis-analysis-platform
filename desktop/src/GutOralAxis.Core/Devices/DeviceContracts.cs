namespace GutOralAxis.Core.Devices;

public sealed record DeviceDescriptor(string Id, string DisplayName, string Transport);

public sealed record DeviceReading(
    string DeviceId,
    DateTimeOffset RecordedAt,
    IReadOnlyDictionary<string, double> Values,
    string Unit);

public interface IDeviceAdapter : IAsyncDisposable
{
    string AdapterName { get; }

    Task<IReadOnlyList<DeviceDescriptor>> DiscoverAsync(CancellationToken cancellationToken);

    Task ConnectAsync(string deviceId, CancellationToken cancellationToken);

    Task<DeviceReading> ReadAsync(CancellationToken cancellationToken);

    Task DisconnectAsync(CancellationToken cancellationToken);
}
