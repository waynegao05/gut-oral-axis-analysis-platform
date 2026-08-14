using System.Text.Json;

namespace GutOralAxis.Infrastructure.Database;

public sealed class AuditRepository(SqliteDatabase database)
{
    public async Task RecordAsync(
        string eventType,
        string outcome,
        string? userId = null,
        string? entityType = null,
        string? entityId = null,
        JsonElement? safeDetails = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(eventType);
        ArgumentException.ThrowIfNullOrWhiteSpace(outcome);
        await using var connection = await database.OpenAsync(cancellationToken).ConfigureAwait(false);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            INSERT INTO audit_logs(
                occurred_utc, user_id, event_type, entity_type, entity_id, outcome, detail_json)
            VALUES($occurredUtc, $userId, $eventType, $entityType, $entityId, $outcome, $detailJson);
            """;
        command.Parameters.AddWithValue("$occurredUtc", DateTimeOffset.UtcNow.ToString("O"));
        command.Parameters.AddWithValue("$userId", (object?)userId ?? DBNull.Value);
        command.Parameters.AddWithValue("$eventType", eventType);
        command.Parameters.AddWithValue("$entityType", (object?)entityType ?? DBNull.Value);
        command.Parameters.AddWithValue("$entityId", (object?)entityId ?? DBNull.Value);
        command.Parameters.AddWithValue("$outcome", outcome);
        command.Parameters.AddWithValue("$detailJson", safeDetails?.GetRawText() ?? "{}");
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }
}
