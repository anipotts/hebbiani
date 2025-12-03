// maximizer.ts
// -----------------------------------------------------------
// A simplified, single-process model of Hebbia's Maximizer.
//
// It shows:
//  - LicenseRequest / LicenseGrant
//  - Model families over providers
//  - Partitioning into pollers
//  - Redis-style leader locks per partition
//  - Token buckets for LLM rate limits
//  - A fake WebSocket gateway with callbacks
//  - A small simulation at the bottom
//
// No AWS, no real Redis needed. Everything is in-memory.
// -----------------------------------------------------------

import crypto from "crypto";

// ---------- Types & basic interfaces -----------------------

type ModelFamilyName = string;
type ProviderName = string;
type ConnectionId = string;

interface LicenseRequest {
  id: string;
  modelFamily: ModelFamilyName;
  tokens: number;
  priority: number; // lower = higher priority
  timestampMs: number;
  connectionId: ConnectionId;
}

interface LicenseGrant {
  requestId: string;
  modelName: ProviderName;
  tokens: number;
  grantedAtMs: number;
  partitionId: number;
}

interface ProviderConfig {
  name: ProviderName;
  tokensPerMinute: number;
}

// ---------- Utility: sleep ---------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------- Token bucket rate limiter ----------------------

class TokenBucket {
  readonly capacity: number;
  readonly refillRatePerSec: number;

  private tokens: number;
  private lastRefillMs: number;

  constructor(capacity: number, refillRatePerSec: number) {
    this.capacity = capacity;
    this.refillRatePerSec = refillRatePerSec;
    this.tokens = capacity;
    this.lastRefillMs = Date.now();
  }

  private refillNow() {
    const now = Date.now();
    const elapsedSec = (now - this.lastRefillMs) / 1000;
    if (elapsedSec <= 0) return;

    const added = elapsedSec * this.refillRatePerSec;
    this.tokens = Math.min(this.capacity, this.tokens + added);
    this.lastRefillMs = now;
  }

  canTake(amount: number): boolean {
    this.refillNow();
    return this.tokens >= amount;
  }

  take(amount: number): boolean {
    this.refillNow();
    if (this.tokens >= amount) {
      this.tokens -= amount;
      return true;
    }
    return false;
  }

  // For scheduling: how long until we *could* take `amount` tokens?
  timeUntilAvailableSec(amount: number): number {
    this.refillNow();
    if (this.tokens >= amount) return 0;

    const missing = amount - this.tokens;
    // if refillRatePerSec is 0, this would be infinite; here we assume > 0
    return missing / this.refillRatePerSec;
  }
}

// ---------- Fake "Redis" lock client -----------------------
//
// In a real distributed system, this would talk to Redis with:
//   SET key value NX PX 5000
//   etc.
// Here it's in-memory and single-process, but the API is identical.
//

interface LockRecord {
  ownerId: string;
  expiresAtMs: number;
}

class InMemoryLockClient {
  private locks: Map<string, LockRecord> = new Map();

  async acquireLock(
    lockName: string,
    ownerId: string,
    ttlMs: number
  ): Promise<boolean> {
    const now = Date.now();
    const existing = this.locks.get(lockName);

    if (!existing || existing.expiresAtMs <= now) {
      this.locks.set(lockName, {
        ownerId,
        expiresAtMs: now + ttlMs,
      });
      return true;
    }

    // lock held by someone else
    return false;
  }

  async renewLock(
    lockName: string,
    ownerId: string,
    ttlMs: number
  ): Promise<boolean> {
    const now = Date.now();
    const existing = this.locks.get(lockName);
    if (!existing) return false;
    if (existing.ownerId !== ownerId) return false;

    existing.expiresAtMs = now + ttlMs;
    this.locks.set(lockName, existing);
    return true;
  }

  async releaseLock(lockName: string, ownerId: string): Promise<void> {
    const existing = this.locks.get(lockName);
    if (!existing) return;
    if (existing.ownerId === ownerId) {
      this.locks.delete(lockName);
    }
  }
}

// ---------- Simple consistent-ish hashing ------------------
//
// For teaching, we’ll use: partitionId = hash(key) % numPartitions.
// Real consistent hashing uses a ring and virtual nodes, but this
// is enough to show the idea.
//

function hashToInt(key: string): number {
  const h = crypto.createHash("sha256").update(key).digest();
  // Take first 4 bytes as unsigned int
  return h.readUInt32BE(0);
}

function choosePartition(key: string, numPartitions: number): number {
  const h = hashToInt(key);
  return h % numPartitions;
}

// ---------- Request queue (DynamoDB-ish) -------------------
//
// We model a simple in-memory priority queue with partitioning.
// In production: this would be DynamoDB with GSI on (modelFamily, partitionId, priority, timestamp).
//

interface QueuedRequest {
  partitionId: number;
  request: LicenseRequest;
}

class RequestQueue {
  private items: QueuedRequest[] = [];

  enqueue(req: LicenseRequest, partitionId: number) {
    this.items.push({ partitionId, request: req });
  }

  // Get the highest priority request for a given family + partition
  // Priority: lower `priority` value is higher priority
  // Ties broken by older timestamp.
  dequeueHighest(
    modelFamily: string,
    partitionId: number
  ): LicenseRequest | undefined {
    let bestIdx = -1;
    for (let i = 0; i < this.items.length; i++) {
      const { partitionId: p, request } = this.items[i];
      if (p !== partitionId) continue;
      if (request.modelFamily !== modelFamily) continue;

      if (bestIdx === -1) {
        bestIdx = i;
        continue;
      }

      const currentBest = this.items[bestIdx].request;
      if (
        request.priority < currentBest.priority ||
        (request.priority === currentBest.priority &&
          request.timestampMs < currentBest.timestampMs)
      ) {
        bestIdx = i;
      }
    }

    if (bestIdx === -1) return undefined;

    const [item] = this.items.splice(bestIdx, 1);
    return item.request;
  }

  // For debugging: how many pending?
  countForPartition(modelFamily: string, partitionId: number): number {
    return this.items.filter(
      (x) =>
        x.partitionId === partitionId && x.request.modelFamily === modelFamily
    ).length;
  }
}

// ---------- Fake WebSocket gateway -------------------------
//
// In production: API Gateway holds the TCP/WebSocket, and the
// backend calls a callback URL to push the LicenseGrant.
// Here: connectionId -> callback function.
//

type GrantCallback = (grant: LicenseGrant) => void;

class FakeGateway {
  private callbacks: Map<ConnectionId, GrantCallback> = new Map();

  registerConnection(connectionId: ConnectionId, cb: GrantCallback) {
    this.callbacks.set(connectionId, cb);
  }

  sendGrant(connectionId: ConnectionId, grant: LicenseGrant) {
    const cb = this.callbacks.get(connectionId);
    if (cb) {
      cb(grant);
    } else {
      console.warn(
        `[Gateway] No callback registered for connection ${connectionId}`
      );
    }
  }
}

// ---------- Model family / Provider runtime state ----------

class ProviderRateLimiter {
  readonly name: ProviderName;
  readonly bucket: TokenBucket;

  constructor(config: ProviderConfig, partitionShare: number) {
    // tokens per second for whole provider:
    const tps = config.tokensPerMinute / 60;
    // each partition gets `partitionShare` fraction of this
    const partitionTps = tps * partitionShare;

    this.name = config.name;
    // Allow a burst capacity of 2x the per-minute rate (i.e., 2 minutes worth),
    // so that short-term spikes or large requests don't get stuck just because
    // they exceed the 1-minute slice.
    this.bucket = new TokenBucket(
      /* capacity */ config.tokensPerMinute * partitionShare * 2,
      /* refillRate */ partitionTps
    );
  }
}

interface PollerOptions {
  modelFamily: ModelFamilyName;
  partitionId: number;
  lockName: string;
  locker: InMemoryLockClient;
  requestQueue: RequestQueue;
  gateway: FakeGateway;
  providers: ProviderRateLimiter[];
  // Polling / lock timings
  lockTtlMs: number;
  idleSleepMs: number;
  backlogSleepMs: number;
}

// ---------- Poller -----------------------------------------
//
// A poller is a loop running on some server which:
//
// 1) Tries to become the leader for its partition (Redis lock)
// 2) If leader, repeatedly:
//    - Pull highest priority request for its partition
//    - Check provider buckets for availability
//    - If enough tokens, consume and send LicenseGrant
//    - If not, sleep until there might be enough tokens
//

class Poller {
  private readonly modelFamily: string;
  private readonly partitionId: number;
  private readonly lockName: string;
  private readonly locker: InMemoryLockClient;
  private readonly queue: RequestQueue;
  private readonly gateway: FakeGateway;
  private readonly providers: ProviderRateLimiter[];

  private readonly lockTtlMs: number;
  private readonly idleSleepMs: number;
  private readonly backlogSleepMs: number;

  private readonly instanceId: string;
  private running = false;

  constructor(opts: PollerOptions) {
    this.modelFamily = opts.modelFamily;
    this.partitionId = opts.partitionId;
    this.lockName = opts.lockName;
    this.locker = opts.locker;
    this.queue = opts.requestQueue;
    this.gateway = opts.gateway;
    this.providers = opts.providers;

    this.lockTtlMs = opts.lockTtlMs;
    this.idleSleepMs = opts.idleSleepMs;
    this.backlogSleepMs = opts.backlogSleepMs;

    this.instanceId = `poller-${this.modelFamily}-${this.partitionId}-${crypto
      .randomBytes(4)
      .toString("hex")}`;
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.loop().catch((err) =>
      console.error(`[Poller ${this.partitionId}] Loop error`, err)
    );
  }

  stop() {
    this.running = false;
  }

  private async loop() {
    console.log(
      `[Poller ${this.partitionId}] Starting main loop as instance ${this.instanceId}`
    );

    while (this.running) {
      const gotLock = await this.locker.acquireLock(
        this.lockName,
        this.instanceId,
        this.lockTtlMs
      );

      if (!gotLock) {
        // Someone else is leader; back off.
        await sleep(this.idleSleepMs);
        continue;
      }

      // We are leader for this partition while we hold the lock.
      try {
        await this.runSchedulingCycle();
        // Renew lock so we don't lose it mid-cycle.
        await this.locker.renewLock(
          this.lockName,
          this.instanceId,
          this.lockTtlMs
        );
      } finally {
        // NOTE: In a real system, they might hold the lock for many cycles
        // and only release it on shutdown; here we release each iteration
        // to make "leader election" more visible.
        await this.locker.releaseLock(this.lockName, this.instanceId);
      }
    }

    console.log(`[Poller ${this.partitionId}] Stopped.`);
  }

  private async runSchedulingCycle() {
    const pendingCount = this.queue.countForPartition(
      this.modelFamily,
      this.partitionId
    );
    if (pendingCount === 0) {
      // Nothing to do
      await sleep(this.idleSleepMs);
      return;
    }

    // Get highest priority request for this partition + family
    const req = this.queue.dequeueHighest(this.modelFamily, this.partitionId);

    if (!req) {
      await sleep(this.idleSleepMs);
      return;
    }

    // Choose provider that can serve this request soonest
    const choice = this.chooseBestProvider(req.tokens);
    if (!choice) {
      // All providers are too constrained; put the request back and wait
      console.log(
        `[Poller ${this.partitionId}] No provider can currently serve request ${req.id}; backing off`
      );
      // In a real system, you'd reinsert into queue or keep it in memory with delay.
      // Here, we'll just wait and then re-enqueue.
      await sleep(this.backlogSleepMs);
      this.queue.enqueue(req, this.partitionId);
      return;
    }

    const { provider, delaySec } = choice;

    if (delaySec > 0) {
      console.log(
        `[Poller ${this.partitionId}] Waiting ${delaySec.toFixed(
          2
        )}s for tokens for request ${req.id} on provider ${provider.name}`
      );
      await sleep(delaySec * 1000);
    }

    const ok = provider.bucket.take(req.tokens);
    if (!ok) {
      // This can happen if other requests consumed tokens in the interim.
      console.log(
        `[Poller ${this.partitionId}] Race: tokens disappeared for request ${req.id}, re-enqueue`
      );
      this.queue.enqueue(req, this.partitionId);
      return;
    }

    const grant: LicenseGrant = {
      requestId: req.id,
      modelName: provider.name,
      tokens: req.tokens,
      grantedAtMs: Date.now(),
      partitionId: this.partitionId,
    };

    console.log(
      `[Poller ${this.partitionId}] Granting request ${req.id} (${req.tokens} tokens) on ${provider.name}`
    );

    this.gateway.sendGrant(req.connectionId, grant);

    // Short sleep to avoid busy loop when backlog is huge.
    await sleep(5);
  }

  private chooseBestProvider(
    tokensNeeded: number
  ): { provider: ProviderRateLimiter; delaySec: number } | null {
    let best: { provider: ProviderRateLimiter; delaySec: number } | null = null;

    for (const p of this.providers) {
      const delaySec = p.bucket.timeUntilAvailableSec(tokensNeeded);

      if (best === null || delaySec < best.delaySec) {
        best = { provider: p, delaySec };
      }
    }

    return best;
  }
}

// ---------- Maximizer manager ------------------------------
//
// This glue is what services would call.
// It:
//  - Registers model families + partitions
//  - Creates pollers
//  - Accepts LicenseRequests
//  - Maps requests to partitions
//

interface ModelFamilyRuntime {
  name: ModelFamilyName;
  numPartitions: number;
  pollers: Poller[];
}

class Maximizer {
  private readonly queue: RequestQueue;
  private readonly gateway: FakeGateway;
  private readonly locker: InMemoryLockClient;
  private readonly families: Map<ModelFamilyName, ModelFamilyRuntime> =
    new Map();

  constructor(
    queue: RequestQueue,
    gateway: FakeGateway,
    locker: InMemoryLockClient
  ) {
    this.queue = queue;
    this.gateway = gateway;
    this.locker = locker;
  }

  // Configure a model family and its providers.
  // Example: "gpt-4o" with 2 providers and 4 partitions.
  registerModelFamily(
    name: ModelFamilyName,
    providers: ProviderConfig[],
    numPartitions: number
  ) {
    if (this.families.has(name)) {
      throw new Error(`Model family ${name} already registered`);
    }

    const pollers: Poller[] = [];
    const partitionShare = 1 / numPartitions;

    for (let pid = 0; pid < numPartitions; pid++) {
      const runtimeProviders = providers.map(
        (cfg) => new ProviderRateLimiter(cfg, partitionShare)
      );

      const poller = new Poller({
        modelFamily: name,
        partitionId: pid,
        lockName: `maximizer:${name}:partition:${pid}`,
        locker: this.locker,
        requestQueue: this.queue,
        gateway: this.gateway,
        providers: runtimeProviders,
        lockTtlMs: 2000,
        idleSleepMs: 100,
        backlogSleepMs: 300,
      });

      pollers.push(poller);
    }

    this.families.set(name, {
      name,
      numPartitions,
      pollers,
    });
  }

  startAllPollers() {
    for (const fam of this.families.values()) {
      for (const p of fam.pollers) {
        p.start();
      }
    }
  }

  // In real life this would be triggered by a WebSocket->REST event.
  createLicenseRequest(
    req: Omit<LicenseRequest, "id" | "timestampMs">
  ): string {
    const family = this.families.get(req.modelFamily);
    if (!family) {
      throw new Error(`Unknown model family ${req.modelFamily}`);
    }

    const fullReq: LicenseRequest = {
      id: `req_${crypto.randomBytes(4).toString("hex")}`,
      modelFamily: req.modelFamily,
      tokens: req.tokens,
      priority: req.priority,
      timestampMs: Date.now(),
      connectionId: req.connectionId,
    };

    const partitionId = choosePartition(fullReq.id, family.numPartitions);

    console.log(
      `[Maximizer] Enqueue request ${fullReq.id} (tokens=${fullReq.tokens}, priority=${fullReq.priority}) -> partition ${partitionId}`
    );

    this.queue.enqueue(fullReq, partitionId);
    return fullReq.id;
  }
}

// ---------- Simulation -------------------------------------
//
// We'll simulate 1 model family ("gpt-4o") with 2 providers:
//   - openai-gpt-4o: 300k tokens/min
//   - azure-gpt-4o-us-east: 300k tokens/min
//
// And 4 partitions, each with 1/4 of capacity.
//
// We'll open a few fake "connections", send LicenseRequests, and watch
// the pollers schedule them.
//

async function main() {
  const queue = new RequestQueue();
  const gateway = new FakeGateway();
  const locker = new InMemoryLockClient();
  const maximizer = new Maximizer(queue, gateway, locker);

  maximizer.registerModelFamily(
    "gpt-4o",
    [
      { name: "openai-gpt-4o", tokensPerMinute: 300_000 },
      { name: "azure-gpt-4o-us-east", tokensPerMinute: 300_000 },
    ],
    /* numPartitions */ 4
  );

  maximizer.startAllPollers();

  // Register some fake "clients" with callbacks
  function registerClient(name: string): ConnectionId {
    const id = `conn_${name}`;
    gateway.registerConnection(id, (grant) => {
      console.log(
        `[Client ${name}] Received grant for request=${grant.requestId}, model=${grant.modelName}, partition=${grant.partitionId}, tokens=${grant.tokens}`
      );
    });
    return id;
  }

  const connA = registerClient("A");
  const connB = registerClient("B");
  const connC = registerClient("C");

  // Submit some requests with different priorities and sizes
  maximizer.createLicenseRequest({
    modelFamily: "gpt-4o",
    tokens: 50_000,
    priority: 1, // highest priority
    connectionId: connA,
  });

  maximizer.createLicenseRequest({
    modelFamily: "gpt-4o",
    tokens: 80_000,
    priority: 2,
    connectionId: connB,
  });

  maximizer.createLicenseRequest({
    modelFamily: "gpt-4o",
    tokens: 20_000,
    priority: 1,
    connectionId: connC,
  });

  // More requests, including a big one that may need to wait.
  for (let i = 0; i < 6; i++) {
    const conn = i % 2 === 0 ? connA : connB;
    maximizer.createLicenseRequest({
      modelFamily: "gpt-4o",
      tokens: 100_000,
      priority: 3,
      connectionId: conn,
    });
  }

  // Let pollers run for a while
  console.log("\n--- Simulation running for ~10 seconds ---\n");
  await sleep(10_000);
  console.log("\n--- Simulation finished ---\n");

  process.exit(0);
}

main().catch((err) => console.error(err));
