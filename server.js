import express from "express";
import multer from "multer";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-cpu";
import { createCanvas, loadImage } from "canvas";
import os from "os";

// --- Absolute Path Setup ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const MODEL_DIR = path.join(__dirname, "model");

const app = express();
app.use(cors());
app.use(express.json());

// --- Global Variables ---
let model;
let labels = [];
let solutions = {};

// --- Multer Upload Handler ---
const upload = multer({ dest: "uploads/" });

// --- Initialization ---
async function initialize() {
  try {
    await tf.setBackend("cpu");
    console.log("✅ TensorFlow backend set to CPU.");

    // Load labels
    const classIndices = JSON.parse(
      fs.readFileSync(path.join(MODEL_DIR, "class_indices.json"))
    );

    // Sort keys and map values correctly
    labels = Object.keys(classIndices)
      .sort((a, b) => a - b)
      .map((key) => classIndices[key]);

    console.log(`✅ Loaded ${labels.length} labels.`);

    // Load disease info (optional)
    const infoPath = path.join(MODEL_DIR, "diseaseInfo.json");
    if (fs.existsSync(infoPath)) {
      solutions = JSON.parse(fs.readFileSync(infoPath));
      console.log("✅ Loaded disease information.");
    } else {
      console.log("ℹ️ No diseaseInfo.json found.");
    }

    // Load model (manual weight loading)
    console.log("🔄 Loading TensorFlow model...");
    const modelJsonPath = path.join(MODEL_DIR, "model.json");
    const modelJson = JSON.parse(fs.readFileSync(modelJsonPath, "utf8"));

    const shardPaths = modelJson.weightsManifest[0].paths;
    console.log(`🔹 Found ${shardPaths.length} weight file(s).`);

    const weightBuffers = shardPaths.map((shardPath) =>
      fs.readFileSync(path.join(MODEL_DIR, shardPath))
    );

    const combinedWeights = Buffer.concat(weightBuffers);

    const modelIOHandler = {
      load: async () => ({
        modelTopology: modelJson.modelTopology,
        weightSpecs: modelJson.weightsManifest[0].weights,
        weightData: combinedWeights.buffer.slice(
          combinedWeights.byteOffset,
          combinedWeights.byteOffset + combinedWeights.byteLength
        ),
      }),
    };

    model = await tf.loadLayersModel(modelIOHandler);
    console.log("✅ Model loaded successfully.");
    console.log("🚀 Server initialized.");

  } catch (err) {
    console.error("❌ Initialization Error:", err);
  }
}

// --- Prediction Endpoint ---
app.post("/predict", upload.single("image"), async (req, res) => {
  console.log("\n📸 Received image for prediction.");

  if (!req.file) {
    return res.status(400).json({ status: "error", message: "No file uploaded." });
  }

  const imgPath = req.file.path;

  try {
    if (!model) {
      throw new Error("Model not loaded yet.");
    }

    console.log("🖼 Processing image...");
    const image = await loadImage(imgPath);
    const canvas = createCanvas(224, 224);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0, 224, 224);

    // -------------------------
    // FIXED NORMALIZATION HERE
    // -------------------------
    const input = tf.browser
      .fromPixels(canvas)
      .toFloat()
      .div(tf.scalar(255.0))  // CORRECT NORMALIZATION
      .expandDims();

    console.log("🤖 Running model prediction...");
    const prediction = model.predict(input);
    const data = await prediction.data();

    const maxIdx = data.indexOf(Math.max(...data));
    const confidence = (data[maxIdx] * 100).toFixed(2);
    const label = labels[maxIdx] || "Unknown";

    console.log(`✅ Prediction: ${label} (${confidence}%)`);

    const info = solutions[label] || {
      disease: label,
      solution: "Use proper fertilizers and care.",
      pesticide: "Apply recommended pesticide.",
    };

    return res.json({
      status: "success",
      label,
      confidence,
      solution: info.solution,
      pesticide: info.pesticide,
    });

  } catch (err) {
    console.error("❌ Prediction Error:", err.message);
    return res.status(500).json({
      status: "error",
      message: err.message || "Prediction failed.",
    });
  } finally {
    // Delete uploaded image
    if (fs.existsSync(imgPath)) {
      fs.unlinkSync(imgPath);
      console.log("🧹 Cleaned up uploaded image.");
    }
  }
});

// --- Start Server ---
const PORT = 5000;
const HOST = "10.133.88.92";

app.listen(PORT, HOST, () => {
  console.log(`🌱 Server running at http://${HOST}:${PORT}`);
  const interfaces = os.networkInterfaces();
  Object.keys(interfaces).forEach((ifaceName) => {
    interfaces[ifaceName].forEach((iface) => {
      if (iface.family === "IPv4" && !iface.internal) {
        console.log(`📱 Connect using: http://${iface.address}:${PORT}`);
      }
    });
  });

  initialize();
});
