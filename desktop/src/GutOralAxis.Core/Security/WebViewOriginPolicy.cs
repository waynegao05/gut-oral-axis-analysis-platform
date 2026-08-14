namespace GutOralAxis.Core.Security;

public static class WebViewOriginPolicy
{
    public static bool IsAllowedApplicationSource(string? source, string applicationHost)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(applicationHost);
        return Uri.TryCreate(source, UriKind.Absolute, out var uri)
            && uri.Scheme == Uri.UriSchemeHttps
            && uri.IsDefaultPort
            && string.IsNullOrEmpty(uri.UserInfo)
            && string.Equals(uri.IdnHost, applicationHost, StringComparison.OrdinalIgnoreCase);
    }
}
