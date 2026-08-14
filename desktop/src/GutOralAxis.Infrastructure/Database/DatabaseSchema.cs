namespace GutOralAxis.Infrastructure.Database;

public static class DatabaseSchema
{
    public const int CurrentVersion = 1;

    public const string MigrationV1 = """
        CREATE TABLE IF NOT EXISTS schema_info (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            version INTEGER NOT NULL,
            applied_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            external_id TEXT NULL,
            age INTEGER NULL CHECK (age IS NULL OR (age >= 18 AND age <= 75)),
            sex TEXT NULL,
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ix_patients_external_id
            ON patients(external_id) WHERE external_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS samples (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
            sample_type TEXT NOT NULL,
            collected_utc TEXT NULL,
            source_device_id TEXT NULL,
            payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
            created_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_samples_patient_id ON samples(patient_id);

        CREATE TABLE IF NOT EXISTS test_results (
            id TEXT PRIMARY KEY,
            sample_id TEXT NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
            result_type TEXT NOT NULL,
            result_json TEXT NOT NULL CHECK (json_valid(result_json)),
            created_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_test_results_sample_id ON test_results(sample_id);

        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
            sample_id TEXT NULL REFERENCES samples(id) ON DELETE SET NULL,
            model_version TEXT NOT NULL,
            risk_score REAL NULL,
            risk_level TEXT NULL,
            result_json TEXT NOT NULL CHECK (json_valid(result_json)),
            created_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_predictions_patient_id ON predictions(patient_id);

        CREATE TABLE IF NOT EXISTS recommendations (
            id TEXT PRIMARY KEY,
            prediction_id TEXT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
            category TEXT NOT NULL,
            priority TEXT NULL,
            recommendation_json TEXT NOT NULL CHECK (json_valid(recommendation_json)),
            created_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_recommendations_prediction_id
            ON recommendations(prediction_id);

        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            patient_id TEXT NULL REFERENCES patients(id) ON DELETE SET NULL,
            prediction_id TEXT NULL REFERENCES predictions(id) ON DELETE SET NULL,
            display_name TEXT NOT NULL,
            relative_path TEXT NOT NULL UNIQUE,
            sha256 TEXT NOT NULL,
            created_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_reports_patient_id ON reports(patient_id);

        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            role TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
            created_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            occurred_utc TEXT NOT NULL,
            user_id TEXT NULL REFERENCES users(id) ON DELETE SET NULL,
            event_type TEXT NOT NULL,
            entity_type TEXT NULL,
            entity_id TEXT NULL,
            outcome TEXT NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(detail_json))
        );
        CREATE INDEX IF NOT EXISTS ix_audit_logs_occurred_utc ON audit_logs(occurred_utc);

        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL CHECK (json_valid(value_json)),
            updated_utc TEXT NOT NULL
        );
        """;
}
