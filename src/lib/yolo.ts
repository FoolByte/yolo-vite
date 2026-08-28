import * as ort from "onnxruntime-web";

export type Detection = {
  classId: number;
  label: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

export type LetterboxInfo = {
  scale: number;
  padX: number;
  padY: number;
};

const MODEL_WIDTH = 640;
const MODEL_HEIGHT = 640;

// Threshold rendah dulu sebelum NMS (praktik umum Ultralytics),
// baru filter lebih ketat setelah NMS kalau perlu.
const RAW_CONFIDENCE_THRESHOLD = 0.25;
const FINAL_CONFIDENCE_THRESHOLD = 0.45;
const IOU_THRESHOLD = 0.45;

const CLASS_NAMES = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

let session: ort.InferenceSession | null = null;
let sessionLoadPromise: Promise<ort.InferenceSession> | null = null;

export async function loadModel() {
  if (session) return session;
  if (!sessionLoadPromise) {
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
    ort.env.wasm.simd = true;

    sessionLoadPromise = ort.InferenceSession.create("/models/yolo11n.onnx", {
      // WebGPU akan otomatis fallback ke wasm kalau tidak didukung browser.
      executionProviders: ["webgpu", "wasm"],
    });
  }
  session = await sessionLoadPromise;
  return session;
}

/**
 * Menggambar `source` ke `canvas` (640x640) dengan letterbox:
 * resize proporsional + padding abu-abu (114,114,114), TIDAK men-distorsi rasio.
 * Mengembalikan info scale & padding untuk unletterbox koordinat hasil deteksi.
 */
export function letterboxToCanvas(
  source: CanvasImageSource,
  srcW: number,
  srcH: number,
  canvas: HTMLCanvasElement,
): LetterboxInfo {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas context not found");

  const scale = Math.min(MODEL_WIDTH / srcW, MODEL_HEIGHT / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = Math.floor((MODEL_WIDTH - newW) / 2);
  const padY = Math.floor((MODEL_HEIGHT - newH) / 2);

  canvas.width = MODEL_WIDTH;
  canvas.height = MODEL_HEIGHT;

  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, MODEL_WIDTH, MODEL_HEIGHT);
  ctx.drawImage(source, 0, 0, srcW, srcH, padX, padY, newW, newH);

  return { scale, padX, padY };
}

function imageToTensor(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas context not found");

  const imageData = ctx.getImageData(0, 0, MODEL_WIDTH, MODEL_HEIGHT);
  const { data } = imageData;
  const size = MODEL_WIDTH * MODEL_HEIGHT;
  const input = new Float32Array(3 * size);

  for (let i = 0; i < size; i++) {
    input[i] = data[i * 4] / 255;
    input[i + size] = data[i * 4 + 1] / 255;
    input[i + size * 2] = data[i * 4 + 2] / 255;
  }

  return new ort.Tensor("float32", input, [1, 3, MODEL_HEIGHT, MODEL_WIDTH]);
}

function calculateIoU(a: Detection, b: Detection) {
  const ax2 = a.x + a.width;
  const ay2 = a.y + a.height;
  const bx2 = b.x + b.width;
  const by2 = b.y + b.height;

  const ix1 = Math.max(a.x, b.x);
  const iy1 = Math.max(a.y, b.y);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);

  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const intersection = iw * ih;

  const areaA = a.width * a.height;
  const areaB = b.width * b.height;
  const union = areaA + areaB - intersection;

  return union === 0 ? 0 : intersection / union;
}

function nonMaximumSuppression(detections: Detection[]) {
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const selected: Detection[] = [];

  while (sorted.length > 0) {
    const current = sorted.shift()!;
    selected.push(current);
    for (let i = sorted.length - 1; i >= 0; i--) {
      const other = sorted[i];
      if (
        current.classId === other.classId &&
        calculateIoU(current, other) > IOU_THRESHOLD
      ) {
        sorted.splice(i, 1);
      }
    }
  }

  return selected;
}

/** Ubah koordinat dari ruang canvas 640x640 (letterboxed) ke ruang gambar asli. */
function unletterbox(d: Detection, info: LetterboxInfo): Detection {
  return {
    ...d,
    x: (d.x - info.padX) / info.scale,
    y: (d.y - info.padY) / info.scale,
    width: d.width / info.scale,
    height: d.height / info.scale,
  };
}

/**
 * Jalankan deteksi. `canvas` harus sudah berisi hasil letterboxToCanvas,
 * dan `letterboxInfo` dari pemanggilan tersebut dipakai untuk unletterbox hasil.
 * Koordinat hasil dikembalikan dalam ruang `srcW x srcH` (gambar/video asli).
 */
export async function detect(
  canvas: HTMLCanvasElement,
  letterboxInfo: LetterboxInfo,
) {
  const model = await loadModel();
  const tensor = imageToTensor(canvas);
  const inputName = model.inputNames[0];

  const outputs = await model.run({ [inputName]: tensor });
  const output = outputs[model.outputNames[0]];
  const data = output.data as Float32Array;

  const detections: Detection[] = [];
  const numPredictions = 8400;
  const numClasses = 80;

  for (let i = 0; i < numPredictions; i++) {
    const x = data[i];
    const y = data[numPredictions + i];
    const width = data[numPredictions * 2 + i];
    const height = data[numPredictions * 3 + i];

    let bestClass = -1;
    let bestScore = 0;
    for (let classId = 0; classId < numClasses; classId++) {
      const score = data[(4 + classId) * numPredictions + i];
      if (score > bestScore) {
        bestScore = score;
        bestClass = classId;
      }
    }

    if (bestScore < RAW_CONFIDENCE_THRESHOLD || bestClass === -1) continue;

    detections.push({
      classId: bestClass,
      label: CLASS_NAMES[bestClass],
      confidence: bestScore,
      x: x - width / 2,
      y: y - height / 2,
      width,
      height,
    });
  }

  const nmsResult = nonMaximumSuppression(detections);

  return nmsResult
    .filter((d) => d.confidence >= FINAL_CONFIDENCE_THRESHOLD)
    .map((d) => unletterbox(d, letterboxInfo));
}
