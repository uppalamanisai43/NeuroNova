"use client";

import { Scan, Loader2 } from "lucide-react";

interface AnalyzeButtonProps {
  onClick: () => void;
  disabled: boolean;
  loading: boolean;
}

export default function AnalyzeButton({ onClick, disabled, loading }: AnalyzeButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className="
        w-full sm:w-auto min-w-[200px]
        flex items-center justify-center gap-3
        px-8 py-4 rounded-2xl font-semibold text-base
        transition-all duration-200
        disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none
        bg-gradient-to-r from-cyan-500 to-cyan-400
        text-slate-900
        hover:from-cyan-400 hover:to-cyan-300
        shadow-[0_0_20px_rgba(0,212,255,0.25)]
        hover:shadow-[0_0_30px_rgba(0,212,255,0.45)]
        active:scale-[0.98]
      "
      aria-label={loading ? "Analyzing MRI…" : "Analyze MRI"}
    >
      {loading ? (
        <>
          <Loader2 className="w-5 h-5 animate-spin" />
          Analyzing…
        </>
      ) : (
        <>
          <Scan className="w-5 h-5" />
          Analyze MRI
        </>
      )}
    </button>
  );
}
