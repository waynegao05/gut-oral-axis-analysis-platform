import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { build } from "esbuild";

const outputDirectory = new URL("../.test-build/", import.meta.url);
const outputDirectoryPath = fileURLToPath(outputDirectory);
await mkdir(outputDirectoryPath, { recursive: true });

await build({
  entryPoints: {
    api: fileURLToPath(new URL("../src/api.ts", import.meta.url)),
    transport: fileURLToPath(new URL("../src/transport.ts", import.meta.url)),
  },
  outdir: outputDirectoryPath,
  outExtension: { ".js": ".mjs" },
  bundle: true,
  platform: "node",
  format: "esm",
  target: "node20",
});

const cacheKey = `?test=${Date.now()}`;
const api = await import(`${new URL("api.mjs", outputDirectory).href}${cacheKey}`);
const transport = await import(
  `${new URL("transport.mjs", outputDirectory).href}${cacheKey}`
);

test("HTTP transport preserves the current Flask routes", async () => {
  const calls = [];
  const fakeFetch = async (url, init) => {
    calls.push({ url, init });
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  const client = new transport.HttpTransport(fakeFetch);

  await client.request("standardize", { sample: 1 });
  await client.request("oralAdenoma.schema");

  assert.equal(calls[0].url, "/standardize");
  assert.equal(calls[0].init.method, "POST");
  assert.equal(calls[0].init.body, JSON.stringify({ sample: 1 }));
  assert.equal(calls[1].url, "/internal/oral-adenoma/schema");
  assert.equal(calls[1].init.method, "GET");
});

test("HTTP transport rejects Windows-only host operations", async () => {
  const client = new transport.HttpTransport(async () => {
    throw new Error("fetch should not run");
  });
  await assert.rejects(client.request("file.openJson"), /Windows 桌面版/);
});

test("API decoder accepts both Flask and local-engine success envelopes", async () => {
  api.setTransportForTesting({
    request: async () => ({ status: 200, payload: { ok: true, value: "flask" } }),
  });
  assert.equal((await api.getJson("oralAdenoma.schema")).value, "flask");

  api.setTransportForTesting({
    request: async () => ({
      status: 200,
      payload: { status: "success", request_id: "req-1", value: "engine" },
    }),
  });
  assert.equal((await api.postJson("analyze", {})).value, "engine");
});

test("API decoder keeps structured local-engine errors", async () => {
  api.setTransportForTesting({
    request: async () => ({
      status: 400,
      payload: {
        status: "error",
        error_code: "INVALID_INPUT",
        message: "请检查输入后重新提交。",
        request_id: "req-2",
      },
    }),
  });

  await assert.rejects(
    api.postJson("analyze", {}),
    (error) => {
      assert.equal(error.name, "ApiError");
      assert.equal(error.status, 400);
      assert.equal(error.errorCode, "INVALID_INPUT");
      assert.equal(error.requestId, "req-2");
      return true;
    },
  );
});

class MockBridge {
  listeners = new Set();
  sent = [];

  postMessage(message) {
    this.sent.push(message);
  }

  addEventListener(type, listener) {
    assert.equal(type, "message");
    this.listeners.add(listener);
  }

  removeEventListener(type, listener) {
    assert.equal(type, "message");
    this.listeners.delete(listener);
  }

  respond(message) {
    for (const listener of this.listeners) {
      listener({ data: message });
    }
  }
}

test("WebView transport correlates responses and ignores unrelated messages", async () => {
  const bridge = new MockBridge();
  const client = new transport.WebViewTransport(bridge, 1000);
  const pending = client.request("analyze", { age: 52 });
  const sent = bridge.sent[0];

  assert.equal(sent.type, "goa.request");
  assert.equal(sent.version, 1);
  assert.equal(sent.operation, "analyze");
  assert.deepEqual(sent.payload, { age: 52 });

  bridge.respond({ type: "unrelated", requestId: sent.requestId });
  bridge.respond({
    type: "goa.response",
    version: 1,
    requestId: sent.requestId,
    status: 200,
    payload: { status: "success" },
  });

  assert.deepEqual(await pending, {
    status: 200,
    payload: { status: "success" },
  });
  client.dispose();
  assert.equal(bridge.listeners.size, 0);
});

test("WebView transport rejects timeouts and oversized messages", async () => {
  const bridge = new MockBridge();
  const timeoutClient = new transport.WebViewTransport(bridge, 5);
  await assert.rejects(
    timeoutClient.request("standardize", {}),
    /请求超时/,
  );
  timeoutClient.dispose();

  const sizeClient = new transport.WebViewTransport(new MockBridge(), 1000, 128);
  await assert.rejects(
    sizeClient.request("analyze", { content: "x".repeat(512) }),
    /数据过大/,
  );
  sizeClient.dispose();
});

test.after(() => {
  api.setTransportForTesting(undefined);
});
