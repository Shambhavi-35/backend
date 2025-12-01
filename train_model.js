import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";
import { createCanvas, loadImage } from "canvas";
import { fileURLToPath } from "url";

// Fix __dirname in ES Modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const IMAGE_SIZE = 224;
const BATCH_SIZE = 32;
const EPOCHS = 10;

const DATA_DIR = path.join(__dirname, "uploads", "PlantVillage");

// Load class folders
const labels = fs.readdirSync(DATA_DIR);
console.log("Detected Classes:", labels);

// Save class index mapping
const classIndices = {};
labels.forEach((label, i) => (classIndices[i] = label));

if (!fs.existsSync("./model")) fs.mkdirSync("./model");
fs.writeFileSync("./model/class_indices.json", JSON.stringify(classIndices, null, 2));

console.log("✔ class_indices.json saved");

// Convert image to tensor
async function loadImageTensor(filePath) {
    const img = await loadImage(filePath);
    const canvas = createCanvas(IMAGE_SIZE, IMAGE_SIZE);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

    return tf.browser.fromPixels(canvas)
        .toFloat()
        .div(tf.scalar(255))
        .expandDims();
}

// Load full dataset
async function loadDataset() {
    const xs = [];
    const ys = [];

    console.log("Loading dataset...");

    for (let i = 0; i < labels.length; i++) {
        const label = labels[i];
        const folder = path.join(DATA_DIR, label);
        const files = fs.readdirSync(folder);

        for (const file of files) {
            const tensor = await loadImageTensor(path.join(folder, file));

            xs.push(tensor);
            ys.push(tf.oneHot(i, labels.length));
        }
    }

    return {
        xs: tf.concat(xs),
        ys: tf.concat(ys)
    };
}

(async () => {
    const { xs, ys } = await loadDataset();

    console.log("Dataset Loaded:");
    console.log("Images:", xs.shape);
    console.log("Labels:", ys.shape);

    console.log("Downloading MobileNetV2...");

    const mobilenet = await tf.loadLayersModel(
        "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v2_1.0_224/model.json"
    );

    console.log("✔ MobileNetV2 Loaded");

    const truncated = tf.model({
        inputs: mobilenet.inputs,
        outputs: mobilenet.getLayer("global_average_pooling2d").output,
    });

    truncated.trainable = false;

    const model = tf.sequential();
    model.add(truncated);
    model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.dense({ units: labels.length, activation: "softmax" }));

    model.compile({
        optimizer: tf.train.adam(0.0001),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    console.log("🚀 Training started...");
    await model.fit(xs, ys, {
        epochs: EPOCHS,
        batchSize: BATCH_SIZE,
        validationSplit: 0.2
    });

    console.log("🎉 Training completed!");

    await model.save("file://model");
    console.log("✔ Model saved inside ./model folder");
})();
