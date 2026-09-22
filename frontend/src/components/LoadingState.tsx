"use client";

import { useEffect, useState } from "react";
import { Brain, Scan, Eye, GitMerge, Sparkles, CheckCircle2 } from "lucide-react";

const STEPS = [
  { icon: Brain,        label: "Preprocessing MRI scan",         detail: "Alpha flattening · Aspect letterbox · Contrast check" },
  { icon: Scan,         label: "Multi-view scan generation",     detail: "Synthesizing test-time augmented perspectives" },
  { icon: Eye,          label: "Deep pathological feature scan", detail: "Analyzing tissue density, ventricles & margins" },
  { icon: GitMerge,     label: "Cross-validating views",         detail: "Averaging multi-view probability distributions" },
  { icon: Sparkles,     label: "Finalising classification",      detail: "Calibrating diagnostic confidence scores" },
];

export default function LoadingState() {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev < STEPS.length - 1 ? prev + 1 : prev));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="glass-card p-8 sm:p-10 text-center animate-fade-in">
      {/* Spinner */}
      <div className="relative mx-auto mb-8 w-20 h-20">
        <div className="absolute inset-0 rounded-full border-4 border-white/5" />
        <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-cyan-400 animate-spin" />
        <div className="absolute inset-2 rounded-full border-4 border-transparent border-t-cyan-500/50 animate-spin [animation-duration:1.5s] [animation-direction:reverse]" />
        <Brain className="absolute inset-0 m-auto w-7 h-7 text-cyan-400" />
      </div>

      <h2 className="text-xl font-semibold text-white mb-1">Analyzing MRI…</h2>
      <p className="text-slate-500 text-sm mb-8">
        Running comprehensive neural network classification pipeline
      </p>

      {/* Steps */}
      <div className="text-left space-y-2 max-w-sm mx-auto">
        {STEPS.map((step, i) => {
          const Icon = step.icon;
          const done    = i < activeStep;
          const current = i === activeStep;
          return (
            <div
              key={i}
              className={`flex items-start gap-3 px-4 py-2.5 rounded-xl transition-all duration-300
                          ${current ? "bg-cyan-500/10 border border-cyan-500/20" : ""}
                          ${done ? "opacity-50" : current ? "opacity-100" : "opacity-30"}`}
            >
              {/* State indicator */}
              <div className="flex-shrink-0 mt-0.5">
                {done ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : current ? (
                  <div className="w-4 h-4 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin" />
                ) : (
                  <div className="w-4 h-4 rounded-full border border-white/20" />
                )}
              </div>

              {/* Text */}
              <div className="min-w-0">
                <p className={`text-sm font-medium ${current ? "text-cyan-300" : done ? "text-slate-400" : "text-slate-600"}`}>
                  {step.label}
                </p>
                {current && (
                  <p className="text-xs text-slate-500 mt-0.5">{step.detail}</p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Note */}
      <p className="mt-8 text-xs text-slate-600">
        Multi-view evaluation ensures robust, high-accuracy inference
      </p>
    </div>
  );
}
