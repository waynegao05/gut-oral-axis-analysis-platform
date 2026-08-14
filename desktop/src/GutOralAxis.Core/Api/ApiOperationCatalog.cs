namespace GutOralAxis.Core.Api;

public sealed record ApiOperationDefinition(
    string Name,
    HttpMethod Method,
    string EnginePath);

public static class ApiOperationCatalog
{
    private static readonly IReadOnlyDictionary<string, ApiOperationDefinition> Operations =
        new Dictionary<string, ApiOperationDefinition>(StringComparer.Ordinal)
        {
            ["standardize"] = new("standardize", HttpMethod.Post, "/api/v1/standardize"),
            ["predict"] = new("predict", HttpMethod.Post, "/api/v1/predict"),
            ["analyze"] = new("analyze", HttpMethod.Post, "/api/v1/analyze"),
            ["oralAdenoma.schema"] = new(
                "oralAdenoma.schema",
                HttpMethod.Get,
                "/api/v1/oral-adenoma/schema"),
            ["oralAdenoma.analyze"] = new(
                "oralAdenoma.analyze",
                HttpMethod.Post,
                "/api/v1/oral-adenoma/analyze"),
        };
    private static readonly IReadOnlyCollection<ApiOperationDefinition> OperationList =
        Operations.Values.ToArray();

    public static IReadOnlyCollection<ApiOperationDefinition> All => OperationList;

    public static bool TryGet(string name, out ApiOperationDefinition definition) =>
        Operations.TryGetValue(name, out definition!);
}
