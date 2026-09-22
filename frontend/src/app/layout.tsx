import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroNova — Brain MRI Classification",
  description:
    "AI-powered brain MRI classification using an ensemble of fine-tuned deep learning models. Educational and research use only.",
  keywords: ["brain MRI", "deep learning", "CNN", "medical AI", "glioma", "tumor classification"],
  authors: [{ name: "NeuroNova Research" }],
  openGraph: {
    title: "NeuroNova — Brain MRI Classification",
    description:
      "AI-powered brain MRI classification — MobileNetV2 + ResNet50 + VGG16 ensemble.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-navy-900 text-slate-100 antialiased">
        {/* Background grid + glow */}
        <div
          aria-hidden="true"
          className="fixed inset-0 pointer-events-none z-0"
          style={{
            backgroundImage:
              "linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px)",
            backgroundSize: "48px 48px",
          }}
        />
        <div
          aria-hidden="true"
          className="fixed inset-0 pointer-events-none z-0"
          style={{
            background:
              "radial-gradient(ellipse at 50% 0%, rgba(0,212,255,0.10) 0%, transparent 65%)",
          }}
        />
        <div className="relative z-10">{children}</div>
      </body>
    </html>
  );
}
