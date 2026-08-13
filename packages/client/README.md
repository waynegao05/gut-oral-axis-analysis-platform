# @waynegao05/gut-oral-axis-client

Typed TypeScript client for the HTTP API exposed by the Gut-Oral Axis Analysis Platform.

The package contains API types and request helpers only. It does not contain datasets,
trained weights, generated outputs, or model inference code.

## Install

Configure the GitHub Packages registry for the package scope:

```ini
@waynegao05:registry=https://npm.pkg.github.com
```

Then install the package:

```bash
npm install @waynegao05/gut-oral-axis-client
```

GitHub Packages may require an npm-compatible GitHub token with `read:packages`
for installation. Do not commit that token to the repository.

## Use

```ts
import { GutOralAxisClient } from "@waynegao05/gut-oral-axis-client";

const client = new GutOralAxisClient({
  baseUrl: "http://127.0.0.1:8765",
});

const result = await client.analyze({
  microbes: {
    Fusobacterium: 0.18,
    Porphyromonas: 0.14,
  },
  clinical: {
    age: 57,
    sex: "Female",
  },
  metabolites: {},
});

console.log(result.risk_result);
```

The client also exposes:

- `standardize(payload)` for `/standardize`
- `analyze(payload)` for `/analyze`
- `getOralAdenomaSchema()` for `/internal/oral-adenoma/schema`
- `analyzeOralAdenoma(payload)` for `/internal/oral-adenoma/analyze`

The oral adenoma endpoint is an internal research endpoint and is not a clinical
diagnostic service. It must be explicitly enabled by the platform operator.

## Runtime

Node.js 20 or a modern browser with the Fetch API is required. A custom Fetch
implementation can be supplied through the client constructor.

## License

MIT
