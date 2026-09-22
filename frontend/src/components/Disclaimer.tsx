"use client";

import { ShieldAlert, Heart } from "lucide-react";

export default function Disclaimer() {
  return (
    <div className="rounded-2xl border border-amber-500/25 bg-amber-500/5 p-6 animate-fade-in">
      <div className="flex gap-3">
        <ShieldAlert className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
        <div>
          <h4 className="text-sm font-semibold text-amber-300 mb-2">
            Educational / Research Use Only
          </h4>
          <p className="text-sm text-amber-200/70 leading-relaxed">
            NeuroNova is an AI-based research and educational demonstration. It is{" "}
            <strong className="text-amber-200">not a medical diagnostic device</strong> and must not
            be used to diagnose, treat, or make clinical decisions. Results must not replace
            evaluation by a qualified healthcare professional or radiologist.
          </p>
        </div>
      </div>
    </div>
  );
}

export function Footer() {
  return (
    <footer className="border-t border-white/8 mt-16 py-8 px-4 text-center text-sm text-slate-600">
      <div className="flex items-center justify-center gap-1.5 mb-2">
        <Heart className="w-3.5 h-3.5 text-cyan-600" />
        <span>
          <span className="text-slate-400 font-medium">NeuroNova</span> — AI Brain MRI Research Demo
        </span>
      </div>
      <p className="text-xs">
        Ensemble of MobileNetV2 · ResNet50 · VGG16 trained models. For educational use only.
      </p>
      <p className="text-xs mt-1 text-slate-700">
        Not a medical device. Not clinically validated. Do not use for diagnosis.
      </p>
    </footer>
  );
}
