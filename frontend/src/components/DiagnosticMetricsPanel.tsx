"use client";

import { DiagnosticMetrics } from "@/types";
import { Zap, Activity, ShieldCheck, AlertTriangle, Scan, CheckCircle, BarChart3 } from "lucide-react";

interface DiagnosticMetricsPanelProps {
  metrics: DiagnosticMetrics;
}

export default function DiagnosticMetricsPanel({ metrics }: DiagnosticMetricsPanelProps) {
  const isTumor = metrics.tumor_detected;
  const tumorProb = metrics.tumor_probability_percent;
  const healthyProb = metrics.healthy_probability_percent;
  const certainty = metrics.certainty_index_percent;
  const latency = metrics.latency_ms ?? 0;
  const scanMeta = metrics.scan_metadata;
  const roi = scanMeta?.roi_coverage_percent ?? 100;

  return (
    <div className="glass-card p-6 sm:p-8 animate-slide-up space-y-6">
      <div className="flex items-center justify-between border-b border-white/8 pb-4">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
            <Activity className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-slate-200">Clinical & Performance Metrics</h3>
            <p className="text-xs text-slate-500">Diagnostic certainty, latency, and image quality metrics</p>
          </div>
        </div>

        {/* Live latency badge */}
        {latency > 0 && (
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 text-xs font-mono">
            <Zap className="w-3 h-3 text-cyan-400" />
            <span>{latency.toFixed(0)} ms latency</span>
          </div>
        )}
      </div>

      {/* Grid of Key Diagnostic Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Metric 1: Pathology Likelihood */}
        <div
          className="p-4 rounded-xl border flex flex-col justify-between"
          style={{
            background: isTumor ? "rgba(244,63,94,0.06)" : "rgba(34,197,94,0.06)",
            borderColor: isTumor ? "rgba(244,63,94,0.25)" : "rgba(34,197,94,0.25)",
          }}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs uppercase tracking-wider font-mono text-slate-400">
              Pathology Status
            </span>
            {isTumor ? (
              <AlertTriangle className="w-4 h-4 text-rose-400" />
            ) : (
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
            )}
          </div>
          <div>
            <p className={`text-xl font-bold ${isTumor ? "text-rose-400" : "text-emerald-400"}`}>
              {isTumor ? "Tumor Detected" : "Healthy / No Tumor"}
            </p>
            <p className="text-xs text-slate-400 mt-1">
              {isTumor ? `${tumorProb.toFixed(1)}% tumor probability` : `${healthyProb.toFixed(1)}% healthy probability`}
            </p>
          </div>
        </div>

        {/* Metric 2: Decision Certainty Score */}
        <div className="p-4 rounded-xl border border-white/10 bg-white/3 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs uppercase tracking-wider font-mono text-slate-400">
              Decision Certainty
            </span>
            <BarChart3 className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="flex items-baseline gap-1.5">
              <span className="text-2xl font-bold text-white">{certainty.toFixed(1)}</span>
              <span className="text-xs text-slate-400">%</span>
            </div>
            <div className="w-full h-1.5 rounded-full bg-white/10 mt-2 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-cyan-300 transition-all duration-700"
                style={{ width: `${Math.min(100, certainty)}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-500 mt-1">Entropy-calibrated confidence</p>
          </div>
        </div>

        {/* Metric 3: Brain ROI & Quality */}
        <div className="p-4 rounded-xl border border-white/10 bg-white/3 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs uppercase tracking-wider font-mono text-slate-400">
              Brain Tissue ROI
            </span>
            <Scan className="w-4 h-4 text-indigo-400" />
          </div>
          <div>
            <div className="flex items-baseline gap-1.5">
              <span className="text-2xl font-bold text-white">{roi.toFixed(1)}</span>
              <span className="text-xs text-slate-400">%</span>
            </div>
            <div className="w-full h-1.5 rounded-full bg-white/10 mt-2 overflow-hidden">
              <div
                className="h-full rounded-full bg-indigo-400 transition-all duration-700"
                style={{ width: `${Math.min(100, roi)}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-500 mt-1">Parenchyma coverage</p>
          </div>
        </div>
      </div>

      {/* Accuracy Pipeline Quality Badges */}
      <div className="pt-2 border-t border-white/5 flex flex-wrap items-center gap-2 text-xs text-slate-400">
        <span className="text-slate-500 font-mono text-[11px] uppercase mr-1">Pipeline Active:</span>
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-white/4 border border-white/10 text-slate-300">
          <CheckCircle className="w-3 h-3 text-cyan-400" />
          Orientation-Preserving TTA
        </span>
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-white/4 border border-white/10 text-slate-300">
          <CheckCircle className="w-3 h-3 text-emerald-400" />
          Full-Field Anatomical Scaling
        </span>
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-white/4 border border-white/10 text-slate-300">
          <CheckCircle className="w-3 h-3 text-indigo-400" />
          Native Backbone Alignment
        </span>
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-white/4 border border-white/10 text-slate-300">
          <CheckCircle className="w-3 h-3 text-amber-400" />
          Pristine Radiometry
        </span>
      </div>
    </div>
  );
}
