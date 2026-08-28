import { useEffect, useRef, useState } from "react";
import {
  detect,
  loadModel,
  letterboxToCanvas,
  type Detection,
} from "../lib/yolo";
import "./Objectdetector.css";

type CameraState = "idle" | "starting" | "active" | "denied";

export function ObjectDetector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const runningRef = useRef(false);
  const detectingRef = useRef(false);

  const [modelReady, setModelReady] = useState(false);
  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [detectionCount, setDetectionCount] = useState(0);
  const [fps, setFps] = useState(0);

  useEffect(() => {
    loadModel()
      .then(() => setModelReady(true))
      .catch((error) => {
        console.error("Gagal load model:", error);
      });
    return () => {
      stopCamera();
    };
  }, []);

  async function startCamera() {
    if (!modelReady || cameraState === "starting") return;
    setCameraState("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCameraState("active");
      runningRef.current = true;
      detectLoop();
    } catch (error) {
      console.error("Gagal membuka kamera:", error);
      setCameraState("denied");
    }
  }

  function stopCamera() {
    runningRef.current = false;
    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
    }
    const video = videoRef.current;
    if (video) {
      const stream = video.srcObject as MediaStream | null;
      stream?.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
    }
    const overlay = overlayRef.current;
    overlay?.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
    setCameraState("idle");
    setDetectionCount(0);
    setFps(0);
  }

  function handleToggle() {
    if (cameraState === "active") {
      stopCamera();
    } else {
      startCamera();
    }
  }

  function drawDetections(results: Detection[]) {
    const video = videoRef.current;
    const canvas = overlayRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return;

    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    for (const detection of results) {
      const { x, y, width: boxWidth, height: boxHeight } = detection;

      ctx.lineWidth = Math.max(2, width / 220);
      ctx.strokeStyle = "#B4FF39";
      ctx.strokeRect(x, y, boxWidth, boxHeight);

      const label = `${detection.label} ${Math.round(
        detection.confidence * 100,
      )}%`;
      const fontSize = Math.max(14, Math.round(width / 45));
      ctx.font = `600 ${fontSize}px -apple-system, system-ui, sans-serif`;
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = "#B4FF39";
      ctx.fillRect(
        x,
        Math.max(0, y - fontSize - 10),
        textWidth + 16,
        fontSize + 10,
      );
      ctx.fillStyle = "#0A0A0A";
      ctx.fillText(label, x + 8, Math.max(fontSize + 2, y - 6));
    }
  }

  async function detectLoop() {
    if (!runningRef.current) return;
    animationRef.current = requestAnimationFrame(detectLoop);
    if (detectingRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

    const srcW = video.videoWidth;
    const srcH = video.videoHeight;
    if (!srcW || !srcH) return;

    detectingRef.current = true;
    try {
      const letterboxInfo = letterboxToCanvas(video, srcW, srcH, canvas);

      const start = performance.now();
      const results = await detect(canvas, letterboxInfo);
      const elapsed = performance.now() - start;

      setFps(Math.round(1000 / elapsed));
      setDetectionCount(results.length);
      drawDetections(results);
    } catch (error) {
      console.error("Detection error:", error);
    } finally {
      detectingRef.current = false;
    }
  }

  const isActive = cameraState === "active";
  const isStarting = cameraState === "starting";

  return (
    <div className="detector-root">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="detector-video"
      />
      <canvas ref={overlayRef} className="detector-overlay" />
      <canvas ref={canvasRef} width={640} height={640} hidden />

      {!isActive && (
        <div className="detector-placeholder">
          {cameraState === "denied" ? (
            <>
              <p className="detector-placeholder-title">Kamera diblokir</p>
              <p className="detector-placeholder-sub">
                Izinkan akses kamera di pengaturan browser, lalu coba lagi
              </p>
            </>
          ) : !modelReady ? (
            <>
              <span className="detector-spinner" />
              <p className="detector-placeholder-sub">Memuat model...</p>
            </>
          ) : (
            <p className="detector-placeholder-sub">
              Tekan tombol untuk mengaktifkan kamera
            </p>
          )}
        </div>
      )}

      <div className="detector-scrim-top" />
      <div className="detector-scrim-bottom" />

      <div className="detector-status">
        <span
          className={`detector-status-dot ${
            isActive ? "detector-status-dot--live" : ""
          }`}
        />
        <span className="detector-status-text">
          {isActive ? "Mendeteksi" : modelReady ? "Siap" : "Memuat"}
        </span>
        {isActive && (
          <>
            <span className="detector-status-sep" />
            <span className="detector-status-mono">{fps} fps</span>
            <span className="detector-status-sep" />
            <span className="detector-status-mono">{detectionCount} objek</span>
          </>
        )}
      </div>

      <div className="detector-controls">
        <button
          type="button"
          className={`detector-shutter ${
            isActive ? "detector-shutter--active" : ""
          } ${isStarting ? "detector-shutter--busy" : ""}`}
          onClick={handleToggle}
          disabled={!modelReady || isStarting}
          aria-label={isActive ? "Nonaktifkan kamera" : "Aktifkan kamera"}
        >
          <span className="detector-shutter-inner" />
        </button>
      </div>
    </div>
  );
}
