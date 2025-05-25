import logging
import os
import requests
from glob import glob

import gradio as gr
from PIL import Image

logging.basicConfig(level=logging.INFO)


def load_best_architectures():
    circuit_paths = sorted(glob("best_architectures/circuit_epoch*.png"))
    loss_paths = sorted(glob("best_architectures/loss_epoch*.png"))

    gallery_items = []
    target_size = (450, 300)

    for circuit_path, loss_path in zip(circuit_paths, loss_paths):
        epoch_id = circuit_path.split("epoch")[-1].split(".")[0]
        metrics_path = f"best_architectures/metrics_epoch{epoch_id}.txt"
        accuracy = "N/A"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                accuracy = f.read().strip()

        circuit_img = Image.open(circuit_path).resize(target_size)
        loss_img = Image.open(loss_path).resize(target_size)
        gallery_items.append((circuit_img, f"Epoch {epoch_id} – Accuracy: {accuracy}"))
        gallery_items.append((loss_img, f"Epoch {epoch_id} – Loss Curve"))

    return gallery_items


def toggle_autoencoder_visibility(dataset_choice):
    visible = dataset_choice == "Digits"
    return gr.update(visible=visible), gr.update(interactive=not visible)


def validate_inputs(dataset, ae_path):
    if dataset == "Digits" and not ae_path.strip():
        return gr.update(interactive=False)
    return gr.update(interactive=True)


with gr.Blocks() as app:
    with gr.Tab("Setup"):
        gr.Markdown("## Quantum Reinforcement Agent Setup")

        dataset_dropdown = gr.Dropdown(
            label="Select Dataset",
            choices=["Iris", "Digits"],
            value="Iris"
        )

        autoencoder_path = gr.Textbox(
            label="Autoencoder Weights Path",
            placeholder="e.g., models/autoencoder_digits.pt",
            visible=False
        )

        gates_checklist = gr.CheckboxGroup(
            label="Select Allowed Gates",
            choices=["rx", "ry", "rz", "cx", "cz"],
            value=["rx", "cz"]
        )

        with gr.Row():
            discount_rate = gr.Slider(0.5, 0.99, value=0.95, step=0.01, label="Discount Rate (γ)")
            learning_rate = gr.Slider(0.0001, 0.1, value=0.01, step=0.0001, label="Learning Rate")

        max_length = gr.Slider(2, 10, value=4, step=1, label="Max Architecture Length")

        with gr.Row():
            start_button = gr.Button("🚀 Start Training", scale=1)
            status_output = gr.Textbox(label="Status", interactive=False, scale=3)

    with gr.Tab("Live Results"):
        gr.Markdown("## 📊 Best Architectures Per Epoch")
        gallery = gr.Gallery(label="Best Architectures per Epoch", columns=2, height="70vh", object_fit="contain")
        timer = gr.Timer(5)
        timer.tick(fn=load_best_architectures, outputs=gallery)

    dataset_dropdown.change(
        fn=toggle_autoencoder_visibility,
        inputs=dataset_dropdown,
        outputs=[autoencoder_path, start_button]
    )

    autoencoder_path.input(
        fn=lambda ae_path: validate_inputs("Digits", ae_path),
        inputs=autoencoder_path,
        outputs=start_button
    )


    def start_training(dataset, gates, gamma, lr, max_length, ae_path):
        if dataset == "Digits" and not ae_path.strip():
            return "❌ Please provide a valid path to the autoencoder weights for the Digits dataset."

        payload = {
            "dataset": dataset,
            "gates": gates,
            "discount": gamma,
            "lr": lr,
            "max_length": max_length,
            "autoencoder_path": ae_path if dataset == "Digits" else None
        }

        try:
            response = requests.post("http://localhost:8000/start-training", json=payload)
            if response.status_code == 200 or response.status_code == 201:
                return f"✅ Job queued: {response.json()}"
            else:
                return f"❌ Server responded with {response.status_code}: {response.text}"
        except Exception as e:
            return f"❌ Failed to send training request: {e}"


    start_button.click(
        fn=start_training,
        inputs=[dataset_dropdown, gates_checklist, discount_rate, learning_rate, max_length, autoencoder_path],
        outputs=status_output
    )

if __name__ == "__main__":
    app.launch(debug=True)
