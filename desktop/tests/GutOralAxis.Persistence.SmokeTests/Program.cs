using System.Security.Cryptography;
using System.Text.Json;
using GutOralAxis.Core.Logging;
using GutOralAxis.Infrastructure.Database;
using GutOralAxis.Infrastructure.Reports;
using Microsoft.Data.Sqlite;

var testRoot = Environment.GetEnvironmentVariable("GOA_TEST_ROOT");
if (string.IsNullOrWhiteSpace(testRoot))
{
    Console.Error.WriteLine("GOA_TEST_ROOT must point to a writable test directory.");
    return 2;
}

testRoot = Path.Combine(Path.GetFullPath(testRoot), Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(testRoot);
var database = new SqliteDatabase(Path.Combine(testRoot, "data", "application.db"), new NullAppLogger());
var tests = new List<(string Name, Func<Task> Run)>
{
    ("schema initialization is idempotent", TestSchemaInitialization),
    ("patient input is constrained and parameterized", TestPatientConstraints),
    ("foreign keys are enforced", TestForeignKeys),
    ("analysis write is atomic", TestAnalysisTransaction),
    ("settings store valid JSON", TestSettings),
    ("reports are contained, hashed, and indexed", TestReports),
    ("audit records contain only safe operation details", TestAudit),
    ("newer database schema is refused", TestFutureSchemaRefusal),
};

var failures = new List<string>();
foreach (var test in tests)
{
    try
    {
        await test.Run();
        Console.WriteLine($"PASS {test.Name}");
    }
    catch (Exception exception)
    {
        failures.Add($"FAIL {test.Name}: {exception}");
    }
}

foreach (var failure in failures)
{
    Console.Error.WriteLine(failure);
}
Console.WriteLine($"{tests.Count - failures.Count}/{tests.Count} persistence smoke tests passed.");
return failures.Count == 0 ? 0 : 1;

async Task TestSchemaInitialization()
{
    await database.InitializeAsync();
    await database.InitializeAsync();
    await using var connection = await database.OpenAsync();
    await using var command = connection.CreateCommand();
    command.CommandText = """
        SELECT COUNT(*) FROM sqlite_master
        WHERE type = 'table' AND name IN (
            'schema_info', 'patients', 'samples', 'test_results', 'predictions',
            'recommendations', 'reports', 'users', 'audit_logs', 'app_settings');
        """;
    Assert(Convert.ToInt32(await command.ExecuteScalarAsync()) == 10, "Expected database tables were not created.");
    command.CommandText = "PRAGMA user_version;";
    Assert(Convert.ToInt32(await command.ExecuteScalarAsync()) == 1, "Schema version was not persisted.");
}

async Task TestPatientConstraints()
{
    var repository = new PatientRepository(database);
    var now = DateTimeOffset.UtcNow;
    var suspiciousId = "patient'; DROP TABLE patients; --";
    await repository.UpsertAsync(new PatientRecord(suspiciousId, "EXTERNAL-1", 52, "Female", now, now));
    var patient = await repository.GetAsync(suspiciousId);
    Assert(patient?.Id == suspiciousId, "Parameterized patient ID did not round-trip.");
    await AssertThrowsAsync<ArgumentOutOfRangeException>(
        () => repository.UpsertAsync(new PatientRecord("too-old", null, 76, null, now, now)));

    await using var connection = await database.OpenAsync();
    await using var command = connection.CreateCommand();
    command.CommandText = """
        INSERT INTO patients(id, age, created_utc, updated_utc)
        VALUES('raw-too-old', 76, $now, $now);
        """;
    command.Parameters.AddWithValue("$now", now.ToString("O"));
    await AssertThrowsAsync<SqliteException>(() => command.ExecuteNonQueryAsync());
}

async Task TestForeignKeys()
{
    await using var connection = await database.OpenAsync();
    await using var command = connection.CreateCommand();
    command.CommandText = """
        INSERT INTO samples(id, patient_id, sample_type, payload_json, created_utc)
        VALUES('orphan-sample', 'missing-patient', 'stool', '{}', $now);
        """;
    command.Parameters.AddWithValue("$now", DateTimeOffset.UtcNow.ToString("O"));
    await AssertThrowsAsync<SqliteException>(() => command.ExecuteNonQueryAsync());
}

async Task TestAnalysisTransaction()
{
    var patientId = "analysis-patient";
    var now = DateTimeOffset.UtcNow;
    await new PatientRepository(database).UpsertAsync(
        new PatientRecord(patientId, null, 46, "Male", now, now));
    var repository = new AnalysisRepository(database);
    var duplicateRecommendationId = "duplicate-recommendation";
    var request = new AnalysisPersistenceRequest(
        patientId,
        "rolled-back-sample",
        "stool",
        now,
        null,
        JsonSerializer.SerializeToElement(new { microbes = new { fusobacterium = 0.2 } }),
        [new TestResultDraft("rolled-back-test", "microbiome", JsonSerializer.SerializeToElement(new { valid = true }))],
        new PredictionDraft(
            "rolled-back-prediction",
            "ac_icam_real_outcome_pfs_v8",
            0.71,
            "high",
            JsonSerializer.SerializeToElement(new { status = "success" })),
        [
            new RecommendationDraft(
                duplicateRecommendationId,
                "review",
                "high",
                JsonSerializer.SerializeToElement(new { text = "first" })),
            new RecommendationDraft(
                duplicateRecommendationId,
                "review",
                "high",
                JsonSerializer.SerializeToElement(new { text = "second" })),
        ],
        now);
    await AssertThrowsAsync<SqliteException>(() => repository.SaveAsync(request));
    Assert(await CountAsync("samples", "id", request.SampleId) == 0, "Failed transaction left a sample row.");
    Assert(await CountAsync("predictions", "id", request.Prediction.Id) == 0, "Failed transaction left a prediction row.");

    var successful = request with
    {
        SampleId = "saved-sample",
        TestResults = [new TestResultDraft("saved-test", "microbiome", JsonSerializer.SerializeToElement(new { valid = true }))],
        Prediction = request.Prediction with { Id = "saved-prediction" },
        Recommendations = [
            new RecommendationDraft(
                "saved-recommendation",
                "review",
                "high",
                JsonSerializer.SerializeToElement(new { text = "review with clinician" })),
        ],
    };
    await repository.SaveAsync(successful);
    Assert(await CountAsync("samples", "id", successful.SampleId) == 1, "Successful transaction missed sample.");
    Assert(await CountAsync("predictions", "id", successful.Prediction.Id) == 1, "Successful transaction missed prediction.");
}

async Task TestSettings()
{
    var settings = new AppSettingsRepository(database);
    await settings.SetAsync("display", JsonSerializer.SerializeToElement(new { language = "zh-CN" }));
    await settings.SetAsync("display", JsonSerializer.SerializeToElement(new { language = "en-US" }));
    var value = await settings.GetAsync("display");
    Assert(value?.GetProperty("language").GetString() == "en-US", "Setting upsert did not persist latest JSON.");
}

async Task TestReports()
{
    var reportRoot = Path.Combine(testRoot, "reports");
    var store = new ReportStore(reportRoot, database, new NullAppLogger());
    var report = JsonSerializer.SerializeToElement(new { patient_id = "not-logged", risk = 0.71 });
    var saved = await store.SaveJsonAsync(report, "../unsafe-report.json");
    Assert(File.Exists(saved.FullPath), "Report file was not created.");
    Assert(Path.GetFullPath(saved.FullPath).StartsWith(Path.GetFullPath(reportRoot), StringComparison.OrdinalIgnoreCase),
        "Report escaped its storage root.");
    var digest = Convert.ToHexStringLower(SHA256.HashData(await File.ReadAllBytesAsync(saved.FullPath)));
    Assert(digest == saved.Sha256, "Report SHA-256 does not match file content.");
    var listed = await store.ListAsync();
    Assert(listed.Any(item => item.Id == saved.Id), "Saved report was not indexed.");
}

async Task TestAudit()
{
    var repository = new AuditRepository(database);
    var details = JsonSerializer.SerializeToElement(new
    {
        operation = "analyze",
        status = 200,
    });
    await repository.RecordAsync(
        "desktop.bridge_operation",
        "success",
        entityType: "operation",
        entityId: "analyze",
        safeDetails: details);

    await using var connection = await database.OpenAsync();
    await using var command = connection.CreateCommand();
    command.CommandText = "SELECT event_type, outcome, entity_id, detail_json FROM audit_logs;";
    await using var reader = await command.ExecuteReaderAsync();
    Assert(await reader.ReadAsync(), "Audit row was not recorded.");
    Assert(reader.GetString(0) == "desktop.bridge_operation", "Audit event type changed.");
    Assert(reader.GetString(1) == "success", "Audit outcome changed.");
    Assert(reader.GetString(2) == "analyze", "Audit operation changed.");
    var storedDetails = reader.GetString(3);
    Assert(storedDetails.Contains("analyze", StringComparison.Ordinal), "Audit details omitted the operation.");
    Assert(!storedDetails.Contains("patient", StringComparison.OrdinalIgnoreCase), "Audit details included patient data.");
}

async Task TestFutureSchemaRefusal()
{
    var path = Path.Combine(testRoot, "future", "application.db");
    var future = new SqliteDatabase(path, new NullAppLogger());
    await future.InitializeAsync();
    await using (var connection = await future.OpenAsync())
    await using (var command = connection.CreateCommand())
    {
        command.CommandText = "PRAGMA user_version = 99;";
        await command.ExecuteNonQueryAsync();
    }
    await AssertThrowsAsync<InvalidOperationException>(() => future.InitializeAsync());
}

async Task<long> CountAsync(string table, string column, string value)
{
    var allowedTables = new HashSet<string>(StringComparer.Ordinal) { "samples", "predictions" };
    var allowedColumns = new HashSet<string>(StringComparer.Ordinal) { "id" };
    Assert(allowedTables.Contains(table) && allowedColumns.Contains(column), "Unsafe test SQL identifier.");
    await using var connection = await database.OpenAsync();
    await using var command = connection.CreateCommand();
    command.CommandText = $"SELECT COUNT(*) FROM {table} WHERE {column} = $value;";
    command.Parameters.AddWithValue("$value", value);
    return Convert.ToInt64(await command.ExecuteScalarAsync());
}

static void Assert(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static async Task AssertThrowsAsync<TException>(Func<Task> action)
    where TException : Exception
{
    try
    {
        await action();
    }
    catch (TException)
    {
        return;
    }
    throw new InvalidOperationException($"Expected {typeof(TException).Name}.");
}
