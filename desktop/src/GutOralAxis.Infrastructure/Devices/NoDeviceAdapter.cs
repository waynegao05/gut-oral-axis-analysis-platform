using GutOralAxis.Core.Devices;

namespace GutOralAxis.Infrastructure.Devices;

public sealed class NoDeviceAdapter : IDeviceAdapter
{
    public string AdapterName => "No device configured";

    public Task<IReadOnlyList<DeviceDescriptor>> DiscoverAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult<IReadOnlyList<DeviceDescriptor>>(Array.Empty<DeviceDescriptor>());
    }

    public Task ConnectAsync(string deviceId, CancellationToken cancellationToken) =>
        Task.FromException(new NotSupportedException("尚未配置真实检测设备协议。"));

    public Task<DeviceReading> ReadAsync(CancellationToken cancellationToken) =>
        Task.FromException<DeviceReading>(new NotSupportedException("尚未配置真实检测设备协议。"));

    public Task DisconnectAsync(CancellationToken cancellationToken) => Task.CompletedTask;

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
