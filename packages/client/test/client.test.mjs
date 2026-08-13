import assert from "node:assert/strict";
import test from "node:test";

import {
  GutOralAxisApiError,
  GutOralAxisClient,
  GutOralAxisRequestError,
} from "../dist/index.js";

test("standardize sends JSON to the normalized endpoint", async () => {
  const calls = [];
  const client = new GutOralAxisClient({
    baseUrl: "https://example.test/api/",
    fetch: async (input, init) => {
      calls.push({ input, init });
      return Response.json({
        ok: true,
        source_format: "canonical",
        standardized_payload: {
          microbes: {},
          clinical: { age: 57, sex: "Female" },
          metabolites: {},
        },
      });
    },
  });

  const result = await client.standardize({
    microbes: {},
    clinical: { age: 57, sex: "Female" },
    metabolites: {},
  });

  assert.equal(result.ok, true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].input, "https://example.test/api/standardize");
  assert.equal(calls[0].init.method, "POST");
  assert.equal(calls[0].init.headers.get("content-type"), "application/json");
  assert.deepEqual(JSON.parse(calls[0].init.body), {
    microbes: {},
    clinical: { age: 57, sex: "Female" },
    metabolites: {},
  });
});

test("oral adenoma schema uses GET without a request body", async () => {
  let capturedInit;
  const client = new GutOralAxisClient({
    fetch: async (_input, init) => {
      capturedInit = init;
      return Response.json({
        ok: true,
        model_release: "oral_adenoma_internal_v3",
        research_only: true,
        input_unit: "percent",
        required_sum_range_percent: [95, 105],
        feature_count: 2,
        feature_ids: ["f1", "f2"],
        taxonomies: ["Fusobacterium", "Porphyromonas"],
        accepted_sample_types: ["oral_swab", "saliva"],
        claim_boundary: "Internal research only.",
      });
    },
  });

  const result = await client.getOralAdenomaSchema();

  assert.equal(result.model_release, "oral_adenoma_internal_v3");
  assert.equal(capturedInit.method, "GET");
  assert.equal(capturedInit.body, undefined);
});

test("API failures preserve status and structured payload", async () => {
  const client = new GutOralAxisClient({
    fetch: async () =>
      Response.json(
        { ok: false, errors: ["age must be between 18 and 75"] },
        { status: 400 },
      ),
  });

  await assert.rejects(
    () =>
      client.analyze({
        microbes: {},
        clinical: { age: -1, sex: "Female" },
        metabolites: {},
      }),
    (error) => {
      assert.ok(error instanceof GutOralAxisApiError);
      assert.equal(error.status, 400);
      assert.equal(error.message, "age must be between 18 and 75");
      assert.deepEqual(error.payload, {
        ok: false,
        errors: ["age must be between 18 and 75"],
      });
      return true;
    },
  );
});

test("non-JSON responses fail with a request error", async () => {
  const client = new GutOralAxisClient({
    fetch: async () => new Response("service unavailable", { status: 503 }),
  });

  await assert.rejects(
    () => client.getOralAdenomaSchema(),
    (error) => {
      assert.ok(error instanceof GutOralAxisRequestError);
      assert.match(error.message, /non-JSON response with HTTP 503/);
      return true;
    },
  );
});
