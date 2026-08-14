namespace GutOralAxis.Core.Messaging;

public interface IBridgeOperationHandler
{
    Task<OperationResult> HandleAsync(
        BridgeRequest request,
        CancellationToken cancellationToken);
}

public sealed class BridgeRouter(IBridgeOperationHandler handler)
{
    public async Task<string> RouteAsync(string message, CancellationToken cancellationToken)
    {
        if (!BridgeRequestParser.TryParse(message, out var request, out var parseError))
        {
            return parseError!.ToJson();
        }

        try
        {
            var result = await handler.HandleAsync(request!, cancellationToken).ConfigureAwait(false);
            return BridgeResponse.FromResult(request!.RequestId, result).ToJson();
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return BridgeResponse.Error(
                request!.RequestId,
                499,
                "REQUEST_CANCELLED",
                "桌面请求已取消。").ToJson();
        }
        catch (Exception)
        {
            return BridgeResponse.Error(
                request!.RequestId,
                500,
                "HOST_ERROR",
                "Windows 宿主处理请求失败，请查看技术日志。").ToJson();
        }
    }
}
