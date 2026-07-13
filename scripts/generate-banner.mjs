import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const width = 1000;
const height = 360;
const plot = { x: 420, y: 26, width: 552, height: 308 };
const domain = { xMin: -2, xMax: 2, yMin: -1, yMax: 3 };
const grid = { columns: 69, rows: 38 };

const rosenbrock = (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2;
const gradient = (x, y) => [
  -2 * (1 - x) - 400 * x * (y - x * x),
  200 * (y - x * x),
];

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const mapX = (x) => plot.x + ((x - domain.xMin) / (domain.xMax - domain.xMin)) * plot.width;
const mapY = (y) => plot.y + (1 - (y - domain.yMin) / (domain.yMax - domain.yMin)) * plot.height;

const palette = [
  [7, 20, 38],
  [14, 116, 144],
  [34, 197, 94],
  [250, 204, 21],
  [251, 113, 133],
];

function colorAt(t) {
  const scaled = clamp(t, 0, 1) * (palette.length - 1);
  const index = Math.min(Math.floor(scaled), palette.length - 2);
  const local = scaled - index;
  const color = palette[index].map((value, channel) =>
    Math.round(value + (palette[index + 1][channel] - value) * local),
  );
  return `rgb(${color.join(",")})`;
}

function sampleLossField() {
  const cells = [];
  const cellWidth = plot.width / grid.columns;
  const cellHeight = plot.height / grid.rows;

  for (let row = 0; row < grid.rows; row += 1) {
    for (let column = 0; column < grid.columns; column += 1) {
      const x = domain.xMin + ((column + 0.5) / grid.columns) * (domain.xMax - domain.xMin);
      const y = domain.yMax - ((row + 0.5) / grid.rows) * (domain.yMax - domain.yMin);
      const normalized = clamp(Math.log1p(rosenbrock(x, y)) / 9.25, 0, 1);
      cells.push(
        `<rect x="${(plot.x + column * cellWidth).toFixed(2)}" y="${(plot.y + row * cellHeight).toFixed(2)}" width="${(cellWidth + 0.35).toFixed(2)}" height="${(cellHeight + 0.35).toFixed(2)}" fill="${colorAt(normalized)}"/>`,
      );
    }
  }

  return cells.join("\n");
}

const optimizationSteps = 5000;

function optimizeWithAdam(steps = optimizationSteps) {
  let x = -1.6;
  let y = 2.8;
  let mx = 0;
  let my = 0;
  let vx = 0;
  let vy = 0;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const learningRate = 0.015;
  const points = [[x, y, rosenbrock(x, y)]];

  for (let step = 1; step <= steps; step += 1) {
    const [gx, gy] = gradient(x, y);
    mx = beta1 * mx + (1 - beta1) * gx;
    my = beta1 * my + (1 - beta1) * gy;
    vx = beta2 * vx + (1 - beta2) * gx * gx;
    vy = beta2 * vy + (1 - beta2) * gy * gy;

    const correctedMx = mx / (1 - beta1 ** step);
    const correctedMy = my / (1 - beta1 ** step);
    const correctedVx = vx / (1 - beta2 ** step);
    const correctedVy = vy / (1 - beta2 ** step);

    x -= learningRate * correctedMx / (Math.sqrt(correctedVx) + 1e-8);
    y -= learningRate * correctedMy / (Math.sqrt(correctedVy) + 1e-8);

    if (step % 40 === 0 || step === steps) {
      points.push([x, y, rosenbrock(x, y)]);
    }
  }

  return points;
}

function trajectoryPath(points) {
  return points
    .map(([x, y], index) => `${index === 0 ? "M" : "L"} ${mapX(x).toFixed(2)} ${mapY(y).toFixed(2)}`)
    .join(" ");
}

function iterationDots(points) {
  return points
    .filter((_, index) => index % 7 === 0)
    .map(([x, y, loss], index) => {
      const radius = index === 0 ? 5 : 2.2;
      const opacity = clamp(0.45 + index * 0.04, 0.45, 0.95);
      return `<circle cx="${mapX(x).toFixed(2)}" cy="${mapY(y).toFixed(2)}" r="${radius}" fill="#f8fafc" fill-opacity="${opacity.toFixed(2)}"><title>iteration ${index * 35}, loss ${loss.toFixed(4)}</title></circle>`;
    })
    .join("\n");
}

const points = optimizeWithAdam();
const pathData = trajectoryPath(points);
const final = points.at(-1);

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="360" viewBox="0 0 1000 360" role="img" aria-labelledby="title desc">
  <title id="title">A real optimization path through the Rosenbrock loss landscape</title>
  <desc id="desc">A code-generated loss field with an Adam optimization trajectory from initialization toward the Rosenbrock minimum.</desc>
  <defs>
    <linearGradient id="panel" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#070d18"/>
      <stop offset="1" stop-color="#14101d"/>
    </linearGradient>
    <filter id="shadow" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="4" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <clipPath id="plotClip"><rect x="${plot.x}" y="${plot.y}" width="${plot.width}" height="${plot.height}" rx="8"/></clipPath>
    <style>
      .sans { font-family: Inter, Arial, sans-serif; }
      .mono { font-family: Consolas, "Courier New", monospace; }
      .descent { stroke-dasharray: 11 8; animation: descend 2.3s linear infinite; }
      .minimum { animation: pulse 2.1s ease-out infinite; transform-origin: ${mapX(1).toFixed(2)}px ${mapY(1).toFixed(2)}px; }
      @keyframes descend { to { stroke-dashoffset: -38; } }
      @keyframes pulse { 0% { opacity: .9; transform: scale(.25); } 80%, 100% { opacity: 0; transform: scale(2.5); } }
    </style>
  </defs>

  <rect width="1000" height="360" rx="14" fill="url(#panel)"/>
  <rect x="1" y="1" width="998" height="358" rx="13" fill="none" stroke="#94a3b8" stroke-opacity="0.18"/>

  <g class="sans">
    <text x="48" y="62" font-size="12" font-weight="700" letter-spacing="2.7" fill="#38bdf8">PREM BABU KANAPARTHI</text>
    <text x="47" y="110" font-size="34" font-weight="650" fill="#f8fafc">Measure the objective.</text>
    <text x="47" y="148" font-size="34" font-weight="650" fill="#f8fafc">Then move the system.</text>

    <text x="48" y="200" class="mono" font-size="14" fill="#94a3b8">f(x,y) = (1-x)^2 + 100(y-x^2)^2</text>
    <text x="48" y="228" class="mono" font-size="14" fill="#cbd5e1">optimizer = Adam</text>
    <text x="48" y="252" class="mono" font-size="14" fill="#cbd5e1">steps = ${optimizationSteps}</text>
    <text x="48" y="276" class="mono" font-size="14" fill="#4ade80">final_loss = ${final[2].toFixed(4)}</text>

    <line x1="48" y1="300" x2="363" y2="300" stroke="#64748b" stroke-opacity="0.35"/>
    <text x="48" y="327" font-size="12" font-weight="700" letter-spacing="1.2" fill="#fb7185">INFERENCE</text>
    <text x="137" y="327" font-size="12" font-weight="700" letter-spacing="1.2" fill="#38bdf8">RETRIEVAL</text>
    <text x="233" y="327" font-size="12" font-weight="700" letter-spacing="1.2" fill="#c084fc">AGENTS</text>
    <text x="299" y="327" font-size="12" font-weight="700" letter-spacing="1.2" fill="#facc15">EVALS</text>
  </g>

  <g clip-path="url(#plotClip)">
${sampleLossField()}
    <rect x="${plot.x}" y="${plot.y}" width="${plot.width}" height="${plot.height}" fill="#020617" fill-opacity="0.16"/>
    <path d="${pathData}" fill="none" stroke="#020617" stroke-opacity="0.7" stroke-width="6" stroke-linejoin="round"/>
    <path class="descent" d="${pathData}" fill="none" stroke="#f8fafc" stroke-width="2.2" stroke-linejoin="round"/>
${iterationDots(points)}
  </g>

  <rect x="${plot.x}" y="${plot.y}" width="${plot.width}" height="${plot.height}" rx="8" fill="none" stroke="#e2e8f0" stroke-opacity="0.3"/>
  <circle class="minimum" cx="${mapX(1).toFixed(2)}" cy="${mapY(1).toFixed(2)}" r="16" fill="none" stroke="#4ade80" stroke-width="2"/>
  <circle cx="${mapX(1).toFixed(2)}" cy="${mapY(1).toFixed(2)}" r="5" fill="#4ade80" filter="url(#shadow)"/>

  <g class="mono" font-size="10" fill="#f8fafc">
    <text x="${(mapX(-1.6) + 9).toFixed(2)}" y="${(mapY(2.8) - 9).toFixed(2)}">x0</text>
    <text x="${(mapX(1) + 10).toFixed(2)}" y="${(mapY(1) + 4).toFixed(2)}">minimum</text>
    <text x="${plot.x + 12}" y="${plot.y + 18}" fill="#e2e8f0">CODE-GENERATED ROSENBROCK LOSS FIELD</text>
  </g>
</svg>`;

const output = fileURLToPath(new URL("../assets/optimization-landscape.svg", import.meta.url));
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, svg, "utf8");

console.log(`Generated ${path.relative(process.cwd(), output)}`);
console.log(`Final point: x=${final[0].toFixed(4)}, y=${final[1].toFixed(4)}, loss=${final[2].toFixed(4)}`);
