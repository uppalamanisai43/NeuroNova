"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, ImageIcon, X, AlertCircle } from "lucide-react";
import { FileInfo } from "@/types";

interface UploadZoneProps {
  fileInfo: FileInfo | null;
  onFileSelect: (info: FileInfo) => void;
  onClear: () => void;
  disabled?: boolean;
}

const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png"];
const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function getImageDimensions(file: File): Promise<{ width: number; height: number }> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => resolve({ width: 0, height: 0 });
    img.src = URL.createObjectURL(file);
  });
}

export default function UploadZone({ fileInfo, onFileSelect, onClear, disabled }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const processFile = useCallback(
    async (file: File) => {
      setValidationError(null);

      if (!ALLOWED_TYPES.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png)$/i)) {
        setValidationError("Unsupported format. Please upload JPG, JPEG, or PNG.");
        return;
      }

      if (file.size > MAX_SIZE_BYTES) {
        setValidationError("Image exceeds the maximum allowed size of 10 MB.");
        return;
      }

      const previewUrl = URL.createObjectURL(file);
      const dimensions = await getImageDimensions(file);
      onFileSelect({ file, previewUrl, dimensions });
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) await processFile(file);
    },
    [processFile]
  );

  const handleInputChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) await processFile(file);
      e.target.value = "";
    },
    [processFile]
  );

  // If a file is already selected, show the preview inside this zone
  if (fileInfo) {
    return (
      <div className="glass-card p-4 sm:p-6 animate-fade-in">
        <div className="flex flex-col sm:flex-row gap-5 items-start">
          {/* Thumbnail */}
          <div className="relative flex-shrink-0 w-full sm:w-48">
            <div className="aspect-square rounded-xl overflow-hidden border border-white/10 bg-white/3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={fileInfo.previewUrl}
                alt="MRI preview"
                className="w-full h-full object-cover"
              />
            </div>
            <button
              onClick={onClear}
              disabled={disabled}
              className="absolute -top-2 -right-2 w-7 h-7 rounded-full bg-slate-700 border border-white/20
                         flex items-center justify-center text-slate-400 hover:text-red-400
                         hover:border-red-400/50 transition-all disabled:opacity-50"
              aria-label="Remove image"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* File info */}
          <div className="flex-1 min-w-0">
            <p className="label-cyan mb-2">MRI Preview</p>
            <p className="text-slate-200 font-medium text-sm truncate mb-1">
              {fileInfo.file.name}
            </p>
            <div className="space-y-1">
              <p className="text-xs text-slate-500">
                <span className="text-slate-400">Size:</span> {formatBytes(fileInfo.file.size)}
              </p>
              {fileInfo.dimensions && fileInfo.dimensions.width > 0 && (
                <p className="text-xs text-slate-500">
                  <span className="text-slate-400">Dimensions:</span>{" "}
                  {fileInfo.dimensions.width} × {fileInfo.dimensions.height} px
                </p>
              )}
              <p className="text-xs text-slate-500">
                <span className="text-slate-400">Format:</span>{" "}
                {fileInfo.file.type || "image/jpeg"}
              </p>
            </div>

            <button
              onClick={() => inputRef.current?.click()}
              disabled={disabled}
              className="mt-4 btn-secondary flex items-center gap-2"
            >
              <Upload className="w-3.5 h-3.5" />
              Replace image
            </button>
          </div>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png"
          className="hidden"
          onChange={handleInputChange}
        />
      </div>
    );
  }

  // Default empty drop zone
  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          relative cursor-pointer rounded-2xl border-2 border-dashed
          transition-all duration-200 p-10 sm:p-14 text-center
          ${dragOver
            ? "border-cyan-400 bg-cyan-500/10 drop-active"
            : "border-white/15 hover:border-cyan-500/40 hover:bg-white/3"
          }
        `}
        role="button"
        aria-label="Upload MRI image"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") inputRef.current?.click(); }}
      >
        {/* Icon */}
        <div
          className={`mx-auto mb-5 w-16 h-16 rounded-2xl flex items-center justify-center
                       transition-all duration-200
                       ${dragOver ? "bg-cyan-500/20 scale-110" : "bg-white/5"}`}
        >
          {dragOver ? (
            <ImageIcon className="w-8 h-8 text-cyan-400" />
          ) : (
            <Upload className="w-8 h-8 text-slate-400" />
          )}
        </div>

        <p className="text-slate-200 font-medium mb-1">
          {dragOver ? "Drop MRI image here" : "Drag & drop your MRI image"}
        </p>
        <p className="text-slate-500 text-sm mb-5">
          or{" "}
          <span className="text-cyan-400 hover:text-cyan-300 underline underline-offset-2">
            browse files
          </span>
        </p>

        <div className="flex flex-wrap justify-center gap-2 text-xs text-slate-600">
          {["JPG", "JPEG", "PNG"].map((ext) => (
            <span key={ext} className="px-2 py-0.5 rounded border border-white/10 font-mono">
              {ext}
            </span>
          ))}
          <span className="px-2 py-0.5 rounded border border-white/10">max 10 MB</span>
        </div>
      </div>

      {/* Validation error */}
      {validationError && (
        <div className="flex items-start gap-2.5 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/30">
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-300">{validationError}</p>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png"
        className="hidden"
        onChange={handleInputChange}
      />
    </div>
  );
}
