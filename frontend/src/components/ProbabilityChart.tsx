"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { PredictionResult, CLASS_DISPLAY, CLASS_COLORS, CLASS_ORDER } from "@/types";
import { BarChart2 } from "lucide-react";

interface ProbabilityChartProps {
  result: PredictionResult;
}

interface TooltipPayload {
  value: number;
  payload: { color: string; name: string };
}

function CustomTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: TooltipPayload[];
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0];
  return (
    <div className="glass-card px-3 py-2 text-sm">
      <p className="font-medium" style={{ color: p.payload.color }}>
        {p.payload.name}
      </p>
      <p className="text-white font-mono">{(p.value * 100).toFixed(2)}%</p>
    </div>
  );
}

// Custom bar shape that reads fill from the datapoint itself
function ColoredBar(props: {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  color?: string;
  isTop?: boolean;
}) {
  const { x = 0, y = 0, width = 0, height = 0, color = "#00d4ff", isTop = false } = props;
  return (
    <rect
      x={x}
      y={y}
      width={width}
      height={height}
      fill={color}
      fillOpacity={isTop ? 1 : 0.45}
      rx={6}
      ry={6}
    />
  );
}

export default function ProbabilityChart({ result }: ProbabilityChartProps) {
  const data = CLASS_ORDER.map((key) => ({
    key,
    name: CLASS_DISPLAY[key] ?? key,
    value: result.probabilities[key],
    color: CLASS_COLORS[key],
    isTop: result.prediction.class === key,
  }));

  // Sort descending by probability
  const sorted = [...data].sort((a, b) => b.value - a.value);

  return (
    <div className="glass-card p-6 sm:p-8 animate-slide-up">
      <div className="flex items-center gap-2 mb-6">
        <BarChart2 className="w-4 h-4 text-cyan-400" />
        <h3 className="section-title mb-0">Class Probabilities</h3>
      </div>

      {/* Recharts bar chart */}
      <div className="h-48 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sorted}
            layout="vertical"
            margin={{ top: 0, right: 48, bottom: 0, left: 0 }}
          >
            <XAxis
              type="number"
              domain={[0, 1]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              tick={{ fontSize: 11, fill: "#64748b" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={90}
              tick={{ fontSize: 12, fill: "#94a3b8" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ fill: "rgba(255,255,255,0.03)" }}
            />
            <Bar
              dataKey="value"
              maxBarSize={24}
              shape={(props: {
                x?: number;
                y?: number;
                width?: number;
                height?: number;
                color?: string;
                isTop?: boolean;
              }) => <ColoredBar {...props} />}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Numeric table with CSS bars */}
      <div className="space-y-2">
        {sorted.map((entry) => (
          <div key={entry.key} className="flex items-center gap-3">
            <span
              className="w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ backgroundColor: entry.color }}
            />
            <span
              className={`flex-1 text-sm ${
                entry.isTop ? "text-slate-200 font-semibold" : "text-slate-400"
              }`}
            >
              {entry.name}
            </span>
            <div className="flex-1 max-w-[120px] sm:max-w-[200px] h-1.5 rounded-full bg-white/5 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${entry.value * 100}%`,
                  backgroundColor: entry.color,
                  opacity: entry.isTop ? 1 : 0.5,
                }}
              />
            </div>
            <span
              className={`font-mono text-sm w-14 text-right ${
                entry.isTop ? "text-white font-bold" : "text-slate-500"
              }`}
            >
              {(entry.value * 100).toFixed(2)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
