"use client";

import { useState, useCallback } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";

import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import UploadZone from "@/components/UploadZone";
import AnalyzeButton from "@/components/AnalyzeButton";
import LoadingState from "@/components/LoadingState";
import PredictionCard from "@/components/PredictionCard";
import ProbabilityChart from "@/components/ProbabilityChart";
import DiagnosticMetricsPanel from "@/components/DiagnosticMetricsPanel";
import Disclaimer, { Footer } from "@/components/Disclaimer";
import { FileInfo, PredictionResult, ApiResponse } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

// ---------------------------------------------------------------------------
// App state machine
// "idle" → "file-selected" → "loading" → "result" | "error"
// ---------------------------------------------------------------------------
type AppState = "idle" | "file-selected" | "loading" | "result" | "error";

export default function HomePage() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [fileInfo, setFileInfo] = useState<FileInfo | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");

  // ── File selected ────────────────────────────────────────────────────────
  const handleFileSelect = useCallback((info: FileInfo) => {
    setFileInfo(info);
    setPrediction(null);
    setErrorMessage("");
    setAppState("file-selected");
  }, []);

  // ── Clear / reset ────────────────────────────────────────────────────────
  const handleClear = useCallback(() => {
    if (fileInfo?.previewUrl) {
      URL.revokeObjectURL(fileInfo.previewUrl);
    }
    setFileInfo(null);
    setPrediction(null);
    setErrorMessage("");
    setAppState("idle");
  }, [fileInfo]);

  // ── Analyze (POST /predict) ───────────────────────────────────────────────
  const handleAnalyze = useCallback(async () => {
    if (!fileInfo) return;

    setAppState("loading");
    setErrorMessage("");
    setPrediction(null);

    const formData = new FormData();
    formData.append("image", fileInfo.file);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      let data: ApiResponse;
      try {
        data = await response.json();
      } catch {
        throw new Error(
          "The server returned an invalid response. Please ensure the backend is running."
        );
      }

      if (!response.ok || !data.success) {
        const msg = !data.success ? data.error : `Server error (${response.status})`;
        setErrorMessage(msg);
        setAppState("error");
        return;
      }

      setPrediction(data as PredictionResult);
      setAppState("result");
    } catch (err: unknown) {
      let msg = "Unable to connect to the NeuroNova AI server. Please make sure the backend is running.";
      if (err instanceof Error && err.message) {
        if (
          err.message.includes("fetch") ||
          err.message.includes("network") ||
          err.message.includes("Failed to fetch")
        ) {
          msg = `Unable to connect to the NeuroNova AI server at ${API_BASE}. Please start the backend (python app.py) and try again.`;
        } else {
          msg = err.message;
        }
      }
      setErrorMessage(msg);
      setAppState("error");
    }
  }, [fileInfo]);

  // ── Retry ────────────────────────────────────────────────────────────────
  const handleRetry = useCallback(() => {
    if (fileInfo) {
      setAppState("file-selected");
      setErrorMessage("");
      setPrediction(null);
    } else {
      handleClear();
    }
  }, [fileInfo, handleClear]);

  // ────────────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-4xl mx-auto w-full px-4 sm:px-6 lg:px-8 pb-12">
        {/* Hero */}
        <HeroSection />

        {/* ── Upload card ─────────────────────────────────────────────────── */}
        {(appState === "idle" || appState === "file-selected") && (
          <section className="space-y-4 mb-8 animate-fade-in">
            <div className="glass-card p-6 sm:p-8 space-y-6">
              <div>
                <p className="label-cyan mb-1">Step 1 — Upload MRI Image</p>
                <p className="text-slate-500 text-sm">
                  Select a brain MRI scan in JPG or PNG format (max 10 MB).
                </p>
              </div>

              <UploadZone
                fileInfo={fileInfo}
                onFileSelect={handleFileSelect}
                onClear={handleClear}
                disabled={false}
              />

              {/* Analyze button */}
              {appState === "file-selected" && (
                <div className="flex flex-col sm:flex-row items-center gap-4 pt-2">
                  <AnalyzeButton
                    onClick={handleAnalyze}
                    disabled={!fileInfo}
                    loading={false}
                  />
                  <p className="text-xs text-slate-600">
                    Image is sent to the local Flask backend for inference. Nothing is uploaded to
                    the internet.
                  </p>
                </div>
              )}
            </div>

            {/* Disclaimer always visible in upload phase */}
            <Disclaimer />
          </section>
        )}

        {/* ── Loading state ────────────────────────────────────────────────── */}
        {appState === "loading" && (
          <section className="mb-8">
            <LoadingState />
          </section>
        )}

        {/* ── Error state ──────────────────────────────────────────────────── */}
        {appState === "error" && (
          <section className="mb-8 animate-fade-in">
            <div className="glass-card p-6 sm:p-8">
              <div className="flex flex-col sm:flex-row items-start gap-4">
                <div className="w-12 h-12 rounded-xl bg-red-500/10 border border-red-500/30 flex items-center justify-center flex-shrink-0">
                  <AlertCircle className="w-6 h-6 text-red-400" />
                </div>
                <div className="flex-1">
                  <h3 className="text-base font-semibold text-red-300 mb-1">Analysis Failed</h3>
                  <p className="text-sm text-slate-400 leading-relaxed mb-4">{errorMessage}</p>
                  <div className="flex flex-wrap gap-3">
                    <button
                      onClick={handleRetry}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 border border-white/15
                                 text-sm text-slate-300 hover:text-white hover:border-white/30 transition-all"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      Try again
                    </button>
                    <button
                      onClick={handleClear}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm text-slate-500
                                 hover:text-slate-300 transition-all"
                    >
                      Upload different image
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ── Results dashboard ────────────────────────────────────────────── */}
        {appState === "result" && prediction && (
          <section className="space-y-6 animate-fade-in">
            {/* Action bar */}
            <div className="flex items-center justify-between">
              <div>
                <p className="label-cyan mb-0.5">Analysis Complete</p>
                <p className="text-slate-500 text-sm">
                  {fileInfo?.file.name ?? "MRI Image"} · AI Brain MRI Classification Report
                </p>
              </div>
              <button
                onClick={handleClear}
                className="btn-secondary flex items-center gap-2"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                New image
              </button>
            </div>

            {/* Main prediction */}
            <PredictionCard result={prediction} />

            {/* Clinical & Performance Metrics Panel */}
            {prediction.metrics && (
              <DiagnosticMetricsPanel metrics={prediction.metrics} />
            )}

            {/* Probabilities chart */}
            <ProbabilityChart result={prediction} />

            {/* Full disclaimer */}
            <Disclaimer />
          </section>
        )}
      </main>

      <Footer />
    </div>
  );
}
