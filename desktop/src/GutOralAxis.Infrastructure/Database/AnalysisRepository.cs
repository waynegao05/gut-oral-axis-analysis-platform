using System.Text.Json;
using Microsoft.Data.Sqlite;

namespace GutOralAxis.Infrastructure.Database;

public sealed record TestResultDraft(string Id, string ResultType, JsonElement Result);

public sealed record PredictionDraft(
    string Id,
    string ModelVersion,
    double? RiskScore,
    string? RiskLevel,
    JsonElement Result);

public sealed record RecommendationDraft(
    string Id,
    string Category,
    string? Priority,
    JsonElement Recommendation);

public sealed record AnalysisPersistenceRequest(
    string PatientId,
    string SampleId,
    string SampleType,
    DateTimeOffset? CollectedUtc,
    string? SourceDeviceId,
    JsonElement SamplePayload,
    IReadOnlyList<TestResultDraft> TestResults,
    PredictionDraft Prediction,
    IReadOnlyList<RecommendationDraft> Recommendations,
    DateTimeOffset CreatedUtc);

public sealed class AnalysisRepository(SqliteDatabase database)
{
    public async Task SaveAsync(
        AnalysisPersistenceRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ValidateRequired(request.PatientId, nameof(request.PatientId));
        ValidateRequired(request.SampleId, nameof(request.SampleId));
        ValidateRequired(request.SampleType, nameof(request.SampleType));
        ValidateRequired(request.Prediction.Id, nameof(request.Prediction.Id));
        ValidateRequired(request.Prediction.ModelVersion, nameof(request.Prediction.ModelVersion));
        ValidateJson(request.SamplePayload, nameof(request.SamplePayload));
        ValidateJson(request.Prediction.Result, nameof(request.Prediction.Result));
        if (request.Prediction.RiskScore is double riskScore && !double.IsFinite(riskScore))
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Risk score must be finite.");
        }

        foreach (var result in request.TestResults)
        {
            ValidateRequired(result.Id, nameof(result.Id));
            ValidateRequired(result.ResultType, nameof(result.ResultType));
            ValidateJson(result.Result, nameof(result.Result));
        }
        foreach (var recommendation in request.Recommendations)
        {
            ValidateRequired(recommendation.Id, nameof(recommendation.Id));
            ValidateRequired(recommendation.Category, nameof(recommendation.Category));
            ValidateJson(recommendation.Recommendation, nameof(recommendation.Recommendation));
        }

        await using var connection = await database.OpenAsync(cancellationToken).ConfigureAwait(false);
        await using var transaction = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken)
            .ConfigureAwait(false);
        try
        {
            await InsertSampleAsync(connection, transaction, request, cancellationToken).ConfigureAwait(false);
            foreach (var result in request.TestResults)
            {
                await InsertTestResultAsync(
                    connection,
                    transaction,
                    request.SampleId,
                    request.CreatedUtc,
                    result,
                    cancellationToken).ConfigureAwait(false);
            }
            await InsertPredictionAsync(connection, transaction, request, cancellationToken).ConfigureAwait(false);
            foreach (var recommendation in request.Recommendations)
            {
                await InsertRecommendationAsync(
                    connection,
                    transaction,
                    request.Prediction.Id,
                    request.CreatedUtc,
                    recommendation,
                    cancellationToken).ConfigureAwait(false);
            }
            await transaction.CommitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            await transaction.RollbackAsync(CancellationToken.None).ConfigureAwait(false);
            throw;
        }
    }

    private static async Task InsertSampleAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        AnalysisPersistenceRequest request,
        CancellationToken cancellationToken)
    {
        await using var command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = """
            INSERT INTO samples(
                id, patient_id, sample_type, collected_utc, source_device_id, payload_json, created_utc)
            VALUES($id, $patientId, $sampleType, $collectedUtc, $sourceDeviceId, $payloadJson, $createdUtc);
            """;
        command.Parameters.AddWithValue("$id", request.SampleId);
        command.Parameters.AddWithValue("$patientId", request.PatientId);
        command.Parameters.AddWithValue("$sampleType", request.SampleType);
        command.Parameters.AddWithValue("$collectedUtc", NullableText(request.CollectedUtc?.ToString("O")));
        command.Parameters.AddWithValue("$sourceDeviceId", NullableText(request.SourceDeviceId));
        command.Parameters.AddWithValue("$payloadJson", request.SamplePayload.GetRawText());
        command.Parameters.AddWithValue("$createdUtc", request.CreatedUtc.ToString("O"));
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    private static async Task InsertTestResultAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        string sampleId,
        DateTimeOffset createdUtc,
        TestResultDraft result,
        CancellationToken cancellationToken)
    {
        await using var command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = """
            INSERT INTO test_results(id, sample_id, result_type, result_json, created_utc)
            VALUES($id, $sampleId, $resultType, $resultJson, $createdUtc);
            """;
        command.Parameters.AddWithValue("$id", result.Id);
        command.Parameters.AddWithValue("$sampleId", sampleId);
        command.Parameters.AddWithValue("$resultType", result.ResultType);
        command.Parameters.AddWithValue("$resultJson", result.Result.GetRawText());
        command.Parameters.AddWithValue("$createdUtc", createdUtc.ToString("O"));
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    private static async Task InsertPredictionAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        AnalysisPersistenceRequest request,
        CancellationToken cancellationToken)
    {
        await using var command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = """
            INSERT INTO predictions(
                id, patient_id, sample_id, model_version, risk_score, risk_level, result_json, created_utc)
            VALUES($id, $patientId, $sampleId, $modelVersion, $riskScore, $riskLevel, $resultJson, $createdUtc);
            """;
        command.Parameters.AddWithValue("$id", request.Prediction.Id);
        command.Parameters.AddWithValue("$patientId", request.PatientId);
        command.Parameters.AddWithValue("$sampleId", request.SampleId);
        command.Parameters.AddWithValue("$modelVersion", request.Prediction.ModelVersion);
        command.Parameters.AddWithValue("$riskScore", NullableValue(request.Prediction.RiskScore));
        command.Parameters.AddWithValue("$riskLevel", NullableText(request.Prediction.RiskLevel));
        command.Parameters.AddWithValue("$resultJson", request.Prediction.Result.GetRawText());
        command.Parameters.AddWithValue("$createdUtc", request.CreatedUtc.ToString("O"));
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    private static async Task InsertRecommendationAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        string predictionId,
        DateTimeOffset createdUtc,
        RecommendationDraft recommendation,
        CancellationToken cancellationToken)
    {
        await using var command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = """
            INSERT INTO recommendations(
                id, prediction_id, category, priority, recommendation_json, created_utc)
            VALUES($id, $predictionId, $category, $priority, $recommendationJson, $createdUtc);
            """;
        command.Parameters.AddWithValue("$id", recommendation.Id);
        command.Parameters.AddWithValue("$predictionId", predictionId);
        command.Parameters.AddWithValue("$category", recommendation.Category);
        command.Parameters.AddWithValue("$priority", NullableText(recommendation.Priority));
        command.Parameters.AddWithValue("$recommendationJson", recommendation.Recommendation.GetRawText());
        command.Parameters.AddWithValue("$createdUtc", createdUtc.ToString("O"));
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    private static object NullableText(string? value) => value is null ? DBNull.Value : value;

    private static object NullableValue<T>(T? value) where T : struct => value.HasValue ? value.Value : DBNull.Value;

    private static void ValidateRequired(string value, string parameterName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException("Value cannot be empty.", parameterName);
        }
    }

    private static void ValidateJson(JsonElement value, string parameterName)
    {
        if (value.ValueKind == JsonValueKind.Undefined)
        {
            throw new ArgumentException("A valid JSON value is required.", parameterName);
        }
    }
}
