namespace GutOralAxis.Core.Security;

public static class PathPolicy
{
    public static string ResolveWithin(string allowedRoot, string relativePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(allowedRoot);
        ArgumentException.ThrowIfNullOrWhiteSpace(relativePath);
        if (Path.IsPathRooted(relativePath))
        {
            throw new InvalidOperationException("Only relative paths are allowed.");
        }

        var root = Path.GetFullPath(allowedRoot);
        var rootWithSeparator = root.EndsWith(Path.DirectorySeparatorChar)
            ? root
            : root + Path.DirectorySeparatorChar;
        var candidate = Path.GetFullPath(Path.Combine(root, relativePath));
        if (!candidate.StartsWith(rootWithSeparator, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("The path leaves the allowed application directory.");
        }

        return candidate;
    }

    public static string SafeFileName(string requestedName, string fallbackExtension)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(fallbackExtension);
        var invalid = Path.GetInvalidFileNameChars();
        var sanitized = new string(
            requestedName
                .Trim()
                .Select(character => invalid.Contains(character) ? '_' : character)
                .ToArray());
        if (string.IsNullOrWhiteSpace(sanitized) || sanitized is "." or "..")
        {
            sanitized = $"report-{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}";
        }

        var extension = fallbackExtension.StartsWith('.')
            ? fallbackExtension
            : $".{fallbackExtension}";
        return Path.HasExtension(sanitized) ? sanitized : sanitized + extension;
    }
}
