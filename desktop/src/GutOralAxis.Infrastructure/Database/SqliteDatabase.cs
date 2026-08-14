using GutOralAxis.Core.Logging;
using Microsoft.Data.Sqlite;

namespace GutOralAxis.Infrastructure.Database;

public sealed class SqliteDatabase(string databasePath, IAppLogger logger)
{
    public string DatabasePath { get; } = Path.GetFullPath(databasePath);

    public string ConnectionString => new SqliteConnectionStringBuilder
    {
        DataSource = DatabasePath,
        Mode = SqliteOpenMode.ReadWriteCreate,
        Cache = SqliteCacheMode.Shared,
        Pooling = true,
    }.ToString();

    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(DatabasePath)!);
        await using var connection = await OpenAsync(cancellationToken).ConfigureAwait(false);
        await using (var versionCommand = connection.CreateCommand())
        {
            versionCommand.CommandText = "PRAGMA user_version;";
            var rawVersion = await versionCommand.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
            var existingVersion = Convert.ToInt32(rawVersion, System.Globalization.CultureInfo.InvariantCulture);
            if (existingVersion > DatabaseSchema.CurrentVersion)
            {
                throw new InvalidOperationException(
                    $"Database schema {existingVersion} is newer than supported schema {DatabaseSchema.CurrentVersion}.");
            }
        }
        await using var transaction = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken)
            .ConfigureAwait(false);
        await using var command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = DatabaseSchema.MigrationV1;
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);

        command.CommandText = """
            INSERT INTO schema_info(singleton, version, applied_utc)
            VALUES(1, $version, $appliedUtc)
            ON CONFLICT(singleton) DO UPDATE SET
                version = excluded.version,
                applied_utc = excluded.applied_utc;
            """;
        command.Parameters.Clear();
        command.Parameters.AddWithValue("$version", DatabaseSchema.CurrentVersion);
        command.Parameters.AddWithValue("$appliedUtc", DateTimeOffset.UtcNow.ToString("O"));
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        command.CommandText = $"PRAGMA user_version = {DatabaseSchema.CurrentVersion};";
        command.Parameters.Clear();
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        await transaction.CommitAsync(cancellationToken).ConfigureAwait(false);
        logger.Information("database.initialized", $"SQLite schema {DatabaseSchema.CurrentVersion} is ready.");
    }

    public async Task<SqliteConnection> OpenAsync(CancellationToken cancellationToken = default)
    {
        var connection = new SqliteConnection(ConnectionString);
        await connection.OpenAsync(cancellationToken).ConfigureAwait(false);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            PRAGMA foreign_keys = ON;
            PRAGMA busy_timeout = 5000;
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            """;
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        return connection;
    }
}
