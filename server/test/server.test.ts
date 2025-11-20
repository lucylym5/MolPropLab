import request from "supertest";
import app from "../src/index";

describe("health", () => {
  it("GET /health", async () => {
    const r = await request(app).get("/health");
    expect(r.status).toBe(200);
    expect(r.body.ok).toBe(true);
  });
});
