import express from "express";
import multer from "multer";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import tf from "@tensorflow/tfjs-node";
import Jimp from "jimp";

// Path setup
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const MODEL_DIR = path.join(__dirname, "model");

const app = express();
app.use(cors());
app.use(express.json());

// Multer for file upload
const upload = multer({ dest: "uploads/" });

// Global variables
let model;
let labels = [];
let solutions = {};

// ------------ INITIALIZE MODEL ------------
async function initialize() {
  try {
    console.log("🔄 Loading model...");

    model = await tf.loadLayersModel(`file://${MODEL_DIR}/model.json`);
    console.log("✅ Model loaded successfully.");

    const classIndices = JSON.parse(
      fs.readFileSync(path.join(MODEL_DIR, "class_indices.json"))
    );

    labels = Object.keys(classIndices)
      .sort((a, b) => a - b)
      .map((key) => classIndices[key]);

    console.log("✅ Labels loaded:", labels.length);

    const infoPath = path.join(MODEL_DIR, "diseaseInfo.json");
    if (fs.existsSync(infoPath)) {
      solutions = JSON.parse(fs.readFileSync(infoPath));
      console.log("✅ Disease info loaded.");
    }

  } catch (error) {
    console.error("❌ Initialization Error:", error);
  }
}

// ------------ PREDICTION API ------------
app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).send("No image uploaded.");

    const image = await Jimp.read(req.file.path);
    image.resize(224, 224);

    const buffer = await image.getBufferAsync(Jimp.MIME_JPEG);

    const input = tf.node
      .decodeImage(buffer)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .div(tf.scalar(255))
      .expandDims();

    const prediction = model.predict(input);
    const data = await prediction.data();

    const maxIdx = data.indexOf(Math.max(...data));
    const confidence = (data[maxIdx] * 100).toFixed(2);
    const label = labels[maxIdx] || "Unknown";

    const info = solutions[label] || {
      solution: "General care needed.",
      pesticide: "Use appropriate pesticide.",
    };

    res.json({
      status: "success",
      label,
      confidence,
      ...info,
    });

  } catch (error) {
    console.error("❌ Prediction Error:", error);
    res.status(500).send("Failed to process image.");
  } finally {
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
  }
});

// ------------ START SERVER ------------
const PORT = process.env.PORT || 5000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Server running on port ${PORT}`);
  initialize();
});
