"use client";

import { PredictionResult, CLASS_DISPLAY } from "@/types";
import { GitMerge, ArrowRight } from "lucide-react";

interface EnsembleExplanationProps {
  result: PredictionResult;
}

export default function EnsembleExplanation({ result }: EnsembleExplanationProps) {
  const { weights } = result.ensemble;
  const { models } = result;

  const modelRows = [
    {
      name: "MobileNetV2",
      weight: weights.mobilenet,
      prediction: CLASS_DISPLAY[models.mobilenet.prediction] ?? models.mobilenet.prediction,
      confidence: models.mobilenet.confidence_percent,
      color: "#818cf8",
      barColor: "bg-violet-500",
    },
    {
      name: "ResNet50",
      weight: weights.resnet50,
      prediction: CLASS_DISPLAY[models.resnet50.prediction] ?? models.resnet50.prediction,
      confidence: models.resnet50.confidence_percent,
      color: "#00d4ff",
      barColor: "bg-cyan-400",
    },
    {
      name: "VGG16",
      weight: weights.vgg16,
      prediction: CLASS_DISPLAY[models.vgg16.prediction] ?? models.vgg16.prediction,
      confidence: models.vgg16.confidence_percent,
      color: "#f59e0b",
      barColor: "bg-amber-400",
    },
  ];

  return (
    <div className="glass-card p-6 sm:p-8 animate-slide-up">
      <div className="flex items-center gap-2 mb-6">
        <GitMerge className="w-4 h-4 text-cyan-400" />
        <h3 className="section-title mb-0">How the Prediction Was Generated</h3>
      </div>

      <p className="text-sm text-slate-400 mb-8 leading-relaxed">
        NeuroNova uses a <strong className="text-slate-200">weighted soft-voting ensemble</strong>: each model
        produces a probability vector over the four classes, and these are combined using the
        weights below. The class with the highest combined probability is the final prediction.
      </p>

      {/* Visual diagram */}
      <div className="flex flex-col sm:flex-row items-stretch gap-4 mb-8">
        {/* Models column */}
        <div className="flex-1 space-y-3">
          {modelRows.map((m) => (
            <div
              key={m.name}
              className="flex items-center gap-3 px-4 py-3 rounded-xl border"
              style={{ borderColor: `${m.color}30`, background: `${m.color}08` }}
            >
              <div className="flex-1">
                <p className="text-sm font-semibold" style={{ color: m.color }}>{m.name}</p>
                <p className="text-xs text-slate-500 mt-0.5">{m.prediction} · {m.confidence.toFixed(1)}%</p>
              </div>
              {/* Weight bar */}
              <div className="flex flex-col items-end gap-1">
                <span className="text-xs font-mono text-slate-400">{(m.weight * 100).toFixed(0)}%</span>
                <div className="w-16 h-1.5 rounded-full bg-white/5 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${m.barColor}`}
                    style={{ width: `${m.weight * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Arrow */}
        <div className="flex sm:flex-col items-center justify-center px-2">
          <div className="flex sm:flex-col items-center gap-1 text-slate-600">
            <div className="hidden sm:block w-px h-10 bg-gradient-to-b from-transparent via-slate-600 to-transparent" />
            <ArrowRight className="w-5 h-5 sm:rotate-90" />
            <div className="hidden sm:block w-px h-10 bg-gradient-to-b from-transparent via-slate-600 to-transparent" />
          </div>
        </div>

        {/* Result column */}
        <div className="flex-1 flex items-center justify-center">
          <div className="w-full px-4 py-5 rounded-xl border border-cyan-500/30 bg-cyan-500/5 text-center">
            <p className="label-cyan mb-2">Weighted Soft Vote</p>
            <p className="text-slate-400 text-xs font-mono mb-3">
              0.2 × MobileNet<br />
              + 0.6 × ResNet50<br />
              + 0.2 × VGG16
            </p>
            <div className="border-t border-white/10 pt-3">
              <p className="text-xs text-slate-500 mb-1">Final Prediction</p>
              <p className="text-lg font-bold text-cyan-400">
                {CLASS_DISPLAY[result.prediction.class] ?? result.prediction.class}
              </p>
              <p className="text-sm text-slate-400">
                {result.prediction.confidence_percent.toFixed(2)}% confidence
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Formula */}
      <div className="rounded-xl bg-white/3 border border-white/8 p-4">
        <p className="label-cyan mb-2">Ensemble Formula</p>
        <code className="text-xs sm:text-sm font-mono text-slate-300 break-all">
          P(class) = 0.20 × P<sub>MobileNet</sub>(class) + 0.60 × P<sub>ResNet50</sub>(class) + 0.20 × P<sub>VGG16</sub>(class)
        </code>
      </div>
    </div>
  );
}
