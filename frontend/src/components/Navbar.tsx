"use client";

import { Brain, FlaskConical } from "lucide-react";

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 w-full border-b border-white/8 bg-navy-900/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-9 h-9 rounded-xl bg-cyan-500/10 border border-cyan-500/30">
            <Brain className="w-5 h-5 text-cyan-400" />
            <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-cyan-400 rounded-full animate-pulse" />
          </div>
          <div>
            <span className="text-lg font-bold tracking-tight text-white">
              Neuro<span className="text-cyan-400">Nova</span>
            </span>
            <p className="text-[10px] text-slate-500 -mt-0.5 uppercase tracking-widest leading-none">
              Brain MRI · AI Research
            </p>
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-emerald-400 font-medium">Research Demo</span>
          </div>
          <div className="flex items-center gap-1.5 text-slate-500 text-xs">
            <FlaskConical className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Educational Use Only</span>
          </div>
        </div>
      </div>
    </nav>
  );
}
