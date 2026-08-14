using Microsoft.Data.Sqlite;

namespace GutOralAxis.Infrastructure.Database;

public sealed record PatientRecord(
    string Id,
    string? ExternalId,
    int? Age,
    string? Sex,
    DateTimeOffset CreatedUtc,
    DateTimeOffset UpdatedUtc);

public sealed class PatientRepository(SqliteDatabase database)
{
    public async Task<PatientRecord> UpsertAsync(
        PatientRecord patient,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(patient.Id);
        if (patient.Age is < 18 or > 75)
        {
            throw new ArgumentOutOfRangeException(nameof(patient), "Age must be between 18 and 75.");
        }

        await using var connection = await database.OpenAsync(cancellationToken).ConfigureAwait(false);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            INSERT INTO patients(id, external_id, age, sex, created_utc, updated_utc)
            VALUES($id, $externalId, $age, $sex, $createdUtc, $updatedUtc)
            ON CONFLICT(id) DO UPDATE SET
                external_id = excluded.external_id,
                age = excluded.age,
                sex = excluded.sex,
                updated_utc = excluded.updated_utc;
            """;
        AddNullable(command, "$externalId", patient.ExternalId);
        AddNullable(command, "$age", patient.Age);
        AddNullable(command, "$sex", patient.Sex);
        command.Parameters.AddWithValue("$id", patient.Id);
        command.Parameters.AddWithValue("$createdUtc", patient.CreatedUtc.ToString("O"));
        command.Parameters.AddWithValue("$updatedUtc", patient.UpdatedUtc.ToString("O"));
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        return patient;
    }

    public async Task<PatientRecord?> GetAsync(
        string id,
        CancellationToken cancellationToken = default)
    {
        await using var connection = await database.OpenAsync(cancellationToken).ConfigureAwait(false);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT id, external_id, age, sex, created_utc, updated_utc
            FROM patients WHERE id = $id;
            """;
        command.Parameters.AddWithValue("$id", id);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        if (!await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            return null;
        }

        return new PatientRecord(
            reader.GetString(0),
            reader.IsDBNull(1) ? null : reader.GetString(1),
            reader.IsDBNull(2) ? null : reader.GetInt32(2),
            reader.IsDBNull(3) ? null : reader.GetString(3),
            DateTimeOffset.Parse(reader.GetString(4)),
            DateTimeOffset.Parse(reader.GetString(5)));
    }

    private static void AddNullable(SqliteCommand command, string name, object? value) =>
        command.Parameters.AddWithValue(name, value ?? DBNull.Value);
}
