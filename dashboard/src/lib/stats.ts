/**
 * Vendored TypeScript port of mrdoob/stats.js (MIT License)
 * Two panels: FPS (0) and Frame Time ms (1)
 */

class StatsPanel {
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  private min = Infinity;
  private max = 0;
  private pr: number;
  private WIDTH: number;
  private HEIGHT: number;
  private TEXT_X: number;
  private TEXT_Y: number;
  private GRAPH_X: number;
  private GRAPH_Y: number;
  private GRAPH_WIDTH: number;
  private GRAPH_HEIGHT: number;
  private name: string;
  private fg: string;
  private bg: string;

  constructor(name: string, fg: string, bg: string) {
    this.name = name;
    this.fg = fg;
    this.bg = bg;
    this.pr = Math.round(window.devicePixelRatio || 1);

    this.WIDTH = 80 * this.pr;
    this.HEIGHT = 48 * this.pr;
    this.TEXT_X = 3 * this.pr;
    this.TEXT_Y = 2 * this.pr;
    this.GRAPH_X = 3 * this.pr;
    this.GRAPH_Y = 15 * this.pr;
    this.GRAPH_WIDTH = 74 * this.pr;
    this.GRAPH_HEIGHT = 30 * this.pr;

    this.canvas = document.createElement("canvas");
    this.canvas.width = this.WIDTH;
    this.canvas.height = this.HEIGHT;
    this.canvas.style.cssText = "width:80px;height:48px";

    this.context = this.canvas.getContext("2d")!;
    this.context.font = `bold ${9 * this.pr}px Helvetica,Arial,sans-serif`;
    this.context.textBaseline = "top";

    this.context.fillStyle = bg;
    this.context.fillRect(0, 0, this.WIDTH, this.HEIGHT);

    this.context.fillStyle = fg;
    this.context.fillText(name, this.TEXT_X, this.TEXT_Y);
    this.context.fillRect(this.GRAPH_X, this.GRAPH_Y, this.GRAPH_WIDTH, this.GRAPH_HEIGHT);

    this.context.fillStyle = bg;
    this.context.globalAlpha = 0.9;
    this.context.fillRect(this.GRAPH_X, this.GRAPH_Y, this.GRAPH_WIDTH, this.GRAPH_HEIGHT);
  }

  update(value: number, maxValue: number) {
    this.min = Math.min(this.min, value);
    this.max = Math.max(this.max, value);

    this.context.fillStyle = this.bg;
    this.context.globalAlpha = 1;
    this.context.fillRect(0, 0, this.WIDTH, this.GRAPH_Y);
    this.context.fillStyle = this.fg;
    this.context.fillText(
      `${Math.round(value)} ${this.name} (${Math.round(this.min)}-${Math.round(this.max)})`,
      this.TEXT_X,
      this.TEXT_Y,
    );

    this.context.drawImage(
      this.canvas,
      this.GRAPH_X + this.pr,
      this.GRAPH_Y,
      this.GRAPH_WIDTH - this.pr,
      this.GRAPH_HEIGHT,
      this.GRAPH_X,
      this.GRAPH_Y,
      this.GRAPH_WIDTH - this.pr,
      this.GRAPH_HEIGHT,
    );

    this.context.fillRect(
      this.GRAPH_X + this.GRAPH_WIDTH - this.pr,
      this.GRAPH_Y,
      this.pr,
      this.GRAPH_HEIGHT,
    );

    this.context.fillStyle = this.bg;
    this.context.globalAlpha = 0.9;
    this.context.fillRect(
      this.GRAPH_X + this.GRAPH_WIDTH - this.pr,
      this.GRAPH_Y,
      this.pr,
      Math.round((1 - value / maxValue) * this.GRAPH_HEIGHT),
    );
  }
}

export class Stats {
  dom: HTMLDivElement;
  private fpsPanel: StatsPanel;
  private msPanel: StatsPanel;
  private beginTime = 0;
  private prevTime = 0;
  private frames = 0;

  constructor() {
    this.dom = document.createElement("div");
    this.dom.style.cssText =
      "position:absolute;top:0;left:0;cursor:pointer;opacity:0.9;z-index:10000";

    this.fpsPanel = new StatsPanel("FPS", "#0ff", "#002");
    this.msPanel = new StatsPanel("MS", "#0f0", "#020");

    this.dom.appendChild(this.fpsPanel.canvas);
    this.dom.appendChild(this.msPanel.canvas);

    this.beginTime = performance.now();
    this.prevTime = this.beginTime;
  }

  begin() {
    this.beginTime = performance.now();
  }

  end() {
    this.frames++;
    const now = performance.now();

    this.msPanel.update(now - this.beginTime, 200);

    if (now >= this.prevTime + 1000) {
      this.fpsPanel.update((this.frames * 1000) / (now - this.prevTime), 100);
      this.prevTime = now;
      this.frames = 0;
    }
  }
}
