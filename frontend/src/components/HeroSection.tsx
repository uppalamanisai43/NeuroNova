"use client";

import { Brain, Sparkles, CheckCircle2 } from "lucide-react";

export default function HeroSection() {
  return (
    <section className="pt-16 pb-10 px-4 text-center">
      {/* Overline tag */}
      <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-cyan-500/30 bg-cyan-500/5 mb-6">
        <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
        <span className="text-xs text-cyan-400 font-medium tracking-wider uppercase">
          AI-Assisted Diagnostics · Multi-View MRI Analysis
        </span>
      </div>

      {/* Main title */}
      <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight mb-4">
        <span className="text-white">Neuro</span>
        <span
          className="text-transparent bg-clip-text"
          style={{
            backgroundImage: "linear-gradient(135deg, #00d4ff 0%, #0891b2 50%, #818cf8 100%)",
          }}
        >
          Nova
        </span>
      </h1>

      <p className="text-xl sm:text-2xl font-light text-slate-400 mb-2">
        AI-Powered Brain MRI Classification
      </p>

      <p className="max-w-2xl mx-auto text-slate-500 text-sm sm:text-base leading-relaxed mb-8">
        Upload a brain MRI scan to instantly classify and analyze potential tumor pathology
        with multi-view test-time augmentation across four diagnostic categories.
      </p>

      {/* Classes badge row */}
      <div className="flex flex-wrap justify-center items-center gap-2.5">
        <span className="text-xs text-slate-500 mr-1 uppercase tracking-wider font-mono">
          Detects:
        </span>
        {[
          { label: "Glioma", desc: "Glial cell tumor", color: "border-rose-500/30 text-rose-400 bg-rose-500/5" },
          { label: "Meningioma", desc: "Meningeal tumor", color: "border-amber-500/30 text-amber-400 bg-amber-500/5" },
          { label: "No Tumor", desc: "Healthy scan", color: "border-emerald-500/30 text-emerald-400 bg-emerald-500/5" },
          { label: "Pituitary", desc: "Pituitary gland tumor", color: "border-indigo-500/30 text-indigo-400 bg-indigo-500/5" },
        ].map((cls) => (
          <div
            key={cls.label}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl border text-xs font-medium ${cls.color}`}
          >
            <CheckCircle2 className="w-3 h-3 opacity-70" />
            <span>{cls.label}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
