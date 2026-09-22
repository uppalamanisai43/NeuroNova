"use client";

import { PredictionResult, CLASS_DISPLAY, ClassProbabilities } from "@/types";
import { Cpu, Layers } from "lucide-react";

interface ModelPredictionCardsProps {
  result: PredictionResult;
}

interface ModelCardProps {
  title: string;
  subtitle: string;
  weight: number;
  prediction: string;
  confidence: number;
  confidencePercent: number;
  probabilities: ClassProbabilities;
  accentColor: string;
  icon: React.ReactNode;
}

function ModelCard({
  title, subtitle, weight, prediction, confidence, confidencePercent, probabilities, accentColor, icon,
}: ModelCardProps) {
  return (
    <div className="glass-card p-5 flex flex-col gap-4 animate-slide-up">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2.5">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: `${accentColor}15`, border: `1px solid ${accentColor}40` }}
          >
            {icon}
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-200">{title}</p>
            <p className="text-xs text-slate-500">{subtitle}</p>
          </div>
        </div>
        {/* Weight badge */}
        <span
          className="text-xs font-mono font-semibold px-2 py-0.5 rounded-md"
          style={{ color: accentColor, background: `${accentColor}15`, border: `1px solid ${accentColor}30` }}
        >
          {(weight * 100).toFixed(0)}% weight
        </span>
      </div>

      {/* Prediction */}
      <div className="text-center py-3 rounded-xl" style={{ background: `${accentColor}0a` }}>
        <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Prediction</p>
        <p className="text-xl font-bold" style={{ color: accentColor }}>
          {CLASS_DISPLAY[prediction] ?? prediction}
        </p>
        <p className="text-slate-400 text-sm mt-1">
          {confidencePercent.toFixed(2)}% confidence
        </p>
      </div>

      {/* Mini probability bars */}
      <div className="space-y-1.5">
        {Object.entries(probabilities)
          .sort((a, b) => b[1] - a[1])
          .map(([cls, prob]) => (
            <div key={cls} className="flex items-center gap-2 text-xs">
              <span className="w-20 text-slate-500 truncate">{CLASS_DISPLAY[cls] ?? cls}</span>
              <div className="flex-1 h-1 rounded-full bg-white/5 overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${prob * 100}%`,
                    backgroundColor: accentColor,
                    opacity: prediction === cls ? 1 : 0.3,
                  }}
                />
              </div>
              <span className="w-11 text-right font-mono text-slate-500">
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
          ))}
      </div>
    </div>
  );
}

export default function ModelPredictionCards({ result }: ModelPredictionCardsProps) {
  const { models } = result;

  return (
    <div className="space-y-4">
      <h3 className="section-title flex items-center gap-2">
        <Layers className="w-4 h-4 text-cyan-400" />
        Individual Model Predictions
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <ModelCard
          title="MobileNetV2"
          subtitle="Lightweight CNN"
          weight={result.ensemble.weights.mobilenet}
          prediction={models.mobilenet.prediction}
          confidence={models.mobilenet.confidence}
          confidencePercent={models.mobilenet.confidence_percent}
          probabilities={models.mobilenet.probabilities}
          accentColor="#818cf8"
          icon={<Cpu className="w-4 h-4 text-violet-400" />}
        />
        <ModelCard
          title="ResNet50"
          subtitle="Deep Residual CNN"
          weight={result.ensemble.weights.resnet50}
          prediction={models.resnet50.prediction}
          confidence={models.resnet50.confidence}
          confidencePercent={models.resnet50.confidence_percent}
          probabilities={models.resnet50.probabilities}
          accentColor="#00d4ff"
          icon={<Cpu className="w-4 h-4 text-cyan-400" />}
        />
        <ModelCard
          title="VGG16"
          subtitle="Very Deep CNN"
          weight={result.ensemble.weights.vgg16}
          prediction={models.vgg16.prediction}
          confidence={models.vgg16.confidence}
          confidencePercent={models.vgg16.confidence_percent}
          probabilities={models.vgg16.probabilities}
          accentColor="#f59e0b"
          icon={<Cpu className="w-4 h-4 text-amber-400" />}
        />
      </div>
    </div>
  );
}
