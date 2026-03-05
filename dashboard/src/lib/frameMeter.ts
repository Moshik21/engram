export interface BenchmarkResult {
  nodeCount: number;
  avgFps: number;
  minFps: number;
  maxFps: number;
  p95FrameTime: number;
  avgFrameTime: number;
  droppedFrames: number;
  sampleCount: number;
}

export class FrameMeter {
  private nodeCount: number;
  private durationMs: number;
  private warmupMs: number;
  private rafId = 0;
  private cancelled = false;

  constructor(nodeCount: number, durationMs = 5000, warmupMs = 2000) {
    this.nodeCount = nodeCount;
    this.durationMs = durationMs;
    this.warmupMs = warmupMs;
  }

  start(): Promise<BenchmarkResult> {
    this.cancelled = false;

    return new Promise((resolve, reject) => {
      const frameTimes: number[] = [];
      let lastTime = 0;
      let warmupDone = false;
      const startTime = performance.now();

      const tick = (now: number) => {
        if (this.cancelled) {
          reject(new Error("Cancelled"));
          return;
        }

        if (lastTime > 0) {
          const dt = now - lastTime;
          const elapsed = now - startTime;

          if (!warmupDone) {
            if (elapsed >= this.warmupMs) warmupDone = true;
          } else {
            frameTimes.push(dt);

            if (elapsed >= this.warmupMs + this.durationMs) {
              resolve(this.compute(frameTimes));
              return;
            }
          }
        }

        lastTime = now;
        this.rafId = requestAnimationFrame(tick);
      };

      this.rafId = requestAnimationFrame(tick);
    });
  }

  cancel() {
    this.cancelled = true;
    cancelAnimationFrame(this.rafId);
  }

  private compute(frameTimes: number[]): BenchmarkResult {
    if (frameTimes.length === 0) {
      return {
        nodeCount: this.nodeCount,
        avgFps: 0,
        minFps: 0,
        maxFps: 0,
        p95FrameTime: 0,
        avgFrameTime: 0,
        droppedFrames: 0,
        sampleCount: 0,
      };
    }

    const sorted = [...frameTimes].sort((a, b) => a - b);
    const sum = sorted.reduce((a, b) => a + b, 0);
    const avg = sum / sorted.length;
    const p95Idx = Math.floor(sorted.length * 0.95);
    const p95 = sorted[p95Idx] ?? sorted[sorted.length - 1];
    const minFrameTime = sorted[0];
    const maxFrameTime = sorted[sorted.length - 1];
    const dropped = frameTimes.filter((t) => t > 33.33).length; // below 30fps

    return {
      nodeCount: this.nodeCount,
      avgFps: 1000 / avg,
      minFps: 1000 / maxFrameTime,
      maxFps: 1000 / minFrameTime,
      p95FrameTime: p95,
      avgFrameTime: avg,
      droppedFrames: dropped,
      sampleCount: sorted.length,
    };
  }
}
