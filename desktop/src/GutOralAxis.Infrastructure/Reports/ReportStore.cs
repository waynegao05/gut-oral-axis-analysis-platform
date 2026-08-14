using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using GutOralAxis.Core.Logging;
using GutOralAxis.Core.Messaging;
using GutOralAxis.Core.Security;
using GutOralAxis.Infrastructure.Database;

namespace GutOralAxis.Infrastructure.Reports;

public sealed record StoredReport(
    string Id,
    string DisplayName,
    string FullPath,
    string Sha256,
    DateTimeOffset CreatedUtc);

public sealed class ReportStore(
    string reportRoot,
    SqliteDatabase database,
    IAppLogger logger)
{
    private readonly string reportRoot = Path.GetFullPath(reportRoot);

    public async Task<StoredReport> SaveJsonAsync(
        JsonElement report,
        string requestedName,
        string? patientId = null,
        string? predictionId = null,
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(reportRoot);
        var id = Guid.NewGuid().ToString("N");
        var createdUtc = DateTimeOffset.UtcNow;
        var safeName = PathPolicy.SafeFileName(requestedName, ".json");
        var relativePath = Path.Combine(createdUtc.ToString("yyyy"), createdUtc.ToString("MM"), $"{id}-{safeName}");
        var fullPath = PathPolicy.ResolveWithin(reportRoot, relativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);

        var json = JsonSerializer.Serialize(report, new JsonSerializerOptions(BridgeJson.Options)
        {
            WriteIndented = true,
        }) + Environment.NewLine;
        var bytes = Encoding.UTF8.GetBytes(json);
        var digest = Convert.ToHexStringLower(SHA256.HashData(bytes));
        var temporaryPath = fullPath + ".tmp";
        await File.WriteAllBytesAsync(temporaryPath, bytes, cancellationToken).ConfigureAwait(false);
        File.Move(temporaryPath, fullPath, overwrite: true);

        try
        {
            await using var connection = await database.OpenAsync(cancellationToken).ConfigureAwait(false);
            await using var command = connection.CreateCommand();
            command.CommandText = """
                INSERT INTO reports(
                    id, patient_id, prediction_id, display_name, relative_path, sha256, created_utc)
                VALUES($id, $patientId, $predictionId, $displayName, $relativePath, $sha256, $createdUtc);
                """;
            command.Parameters.AddWithValue("$id", id);
            command.Parameters.AddWithValue("$patientId", (object?)patientId ?? DBNull.Value);
            command.Parameters.AddWithValue("$predictionId", (object?)predictionId ?? DBNull.Value);
            command.Parameters.AddWithValue("$displayName", safeName);
            command.Parameters.AddWithValue("$relativePath", relativePath.Replace('\\', '/'));
            command.Parameters.AddWithValue("$sha256", digest);
            command.Parameters.AddWithValue("$createdUtc", createdUtc.ToString("O"));
            await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            try
            {
                File.Delete(fullPath);
            }
            catch (IOException)
            {
            }
            throw;
        }

        logger.Information("report.saved", $"Structured report saved. report_id={id}");
        return new StoredReport(id, safeName, fullPath, digest, createdUtc);
    }

    public async Task<IReadOnlyList<StoredReport>> ListAsync(
        int limit = 100,
        CancellationToken cancellationToken = default)
    {
        if (limit is < 1 or > 1_000)
        {
            throw new ArgumentOutOfRangeException(nameof(limit));
        }

        await using var connection = await database.OpenAsync(cancellationToken).ConfigureAwait(false);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT id, display_name, relative_path, sha256, created_utc
            FROM reports ORDER BY created_utc DESC LIMIT $limit;
            """;
        command.Parameters.AddWithValue("$limit", limit);
        var reports = new List<StoredReport>();
        await using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            var relativePath = reader.GetString(2).Replace('/', Path.DirectorySeparatorChar);
            reports.Add(new StoredReport(
                reader.GetString(0),
                reader.GetString(1),
                PathPolicy.ResolveWithin(reportRoot, relativePath),
                reader.GetString(3),
                DateTimeOffset.Parse(reader.GetString(4))));
        }
        return reports;
    }
}
