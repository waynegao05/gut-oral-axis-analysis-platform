using System.Text;
using System.Text.RegularExpressions;
using GutOralAxis.Core.Logging;
using GutOralAxis.Core.Security;

namespace GutOralAxis.Infrastructure.Logging;

public sealed partial class RollingFileLogger : IAppLogger, IDisposable
{
    private readonly string directory;
    private readonly object gate = new();
    private readonly int retentionDays;
    private readonly SensitiveDataRedactor redactor = new();
    private DateOnly currentDate;
    private StreamWriter? writer;

    public RollingFileLogger(string directory, int retentionDays = 30)
    {
        if (retentionDays < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(retentionDays));
        }

        this.directory = Path.GetFullPath(directory);
        this.retentionDays = retentionDays;
        Directory.CreateDirectory(this.directory);
        DeleteExpiredLogs();
    }

    public void Information(string eventName, string message) =>
        Write("INF", eventName, message, null);

    public void Warning(string eventName, string message) =>
        Write("WRN", eventName, message, null);

    public void Error(string eventName, string message, Exception? exception = null) =>
        Write("ERR", eventName, message, exception);

    public void Dispose()
    {
        lock (gate)
        {
            writer?.Dispose();
            writer = null;
        }
    }

    private void Write(string level, string eventName, string message, Exception? exception)
    {
        var now = DateTimeOffset.UtcNow;
        var safeEvent = Normalize(eventName, 80);
        var safeMessage = Normalize(Redact(message), 2_000);
        lock (gate)
        {
            EnsureWriter(DateOnly.FromDateTime(now.UtcDateTime));
            writer!.Write(now.ToString("O"));
            writer.Write(' ');
            writer.Write(level);
            writer.Write(' ');
            writer.Write(safeEvent);
            writer.Write(' ');
            writer.WriteLine(safeMessage);
            if (exception is not null)
            {
                writer.WriteLine(Normalize(exception.ToString(), 8_000));
            }
            writer.Flush();
        }
    }

    private void EnsureWriter(DateOnly date)
    {
        if (writer is not null && currentDate == date)
        {
            return;
        }

        writer?.Dispose();
        currentDate = date;
        var path = Path.Combine(directory, $"desktop-{date:yyyyMMdd}.log");
        writer = new StreamWriter(
            new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read),
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
    }

    private void DeleteExpiredLogs()
    {
        var cutoff = DateTime.UtcNow.Date.AddDays(-retentionDays);
        foreach (var path in Directory.EnumerateFiles(directory, "desktop-*.log"))
        {
            try
            {
                if (File.GetLastWriteTimeUtc(path) < cutoff)
                {
                    File.Delete(path);
                }
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }
        }
    }

    private static string Normalize(string value, int maximumLength)
    {
        var singleLine = value.Replace('\r', ' ').Replace('\n', ' ').Trim();
        return singleLine.Length <= maximumLength
            ? singleLine
            : singleLine[..maximumLength] + "...";
    }

    private string Redact(string value)
    {
        var trimmed = value.TrimStart();
        if (trimmed.StartsWith('{') || trimmed.StartsWith('['))
        {
            return redactor.RedactJson(value);
        }

        return SensitiveAssignmentPattern().Replace(
            value,
            match => $"{match.Groups[1].Value}=[REDACTED]");
    }

    [GeneratedRegex(
        @"(?i)\b(patient_id|patientId|name|phone|email|address|current_medications|drug_allergies)\s*[:=]\s*[^,;\s]+",
        RegexOptions.CultureInvariant)]
    private static partial Regex SensitiveAssignmentPattern();
}
