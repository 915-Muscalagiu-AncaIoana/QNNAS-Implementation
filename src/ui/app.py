import logging
import os
import requests
from glob import glob

import gradio as gr
from PIL import Image

logging.basicConfig(level=logging.INFO)


def load_best_architectures(session_id=None):
    base_path = "best_architectures"
    if session_id:
        base_path = os.path.join(base_path, str(session_id))
    print(session_id)
    circuit_paths = sorted(glob(os.path.join(base_path, "circuit_epoch*.png")))
    loss_paths = sorted(glob(os.path.join(base_path, "loss_epoch*.png")))

    gallery_items = []
    target_size = (450, 300)
    print(circuit_paths)
    print(loss_paths)
    for circuit_path, loss_path in zip(circuit_paths, loss_paths):
        print(circuit_path)
        print(loss_path)
        epoch_id = circuit_path.split("epoch")[-1].split(".")[0]
        metrics_path = os.path.join(base_path, f"metrics_epoch{epoch_id}.txt")
        accuracy = "N/A"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                accuracy = f.read().strip()

        circuit_img = Image.open(circuit_path).resize(target_size)
        loss_img = Image.open(loss_path).resize(target_size)
        gallery_items.append((circuit_img, f"Epoch {epoch_id} – Accuracy: {accuracy}"))
        gallery_items.append((loss_img, f"Epoch {epoch_id} – Loss Curve"))

    return gallery_items


def load_sessions():
    try:
        response = requests.get("http://localhost:8000/sessions/")
        print(response)
        if response.ok:
            sessions = response.json()
            return [f"{s['id']} – {s['dataset']} – {s['status']}" for s in sessions]
        return []
    except Exception as e:
        logging.warning(f"Failed to load sessions: {e}")
        return []


def toggle_autoencoder_visibility(dataset_choice):
    visible = dataset_choice == "Digits"
    return gr.update(visible=visible), gr.update(interactive=not visible)


def validate_inputs(dataset, ae_path):
    if dataset == "Digits" and not ae_path.strip():
        return gr.update(interactive=False)
    return gr.update(interactive=True)


def extract_session_id(session_label):
    return session_label.split(" – ")[0] if session_label else None


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
            value=["rx", "ry", "rz", "cx", "cz"]
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

        gallery = gr.Gallery(
            label="Best Architectures per Epoch",
            columns=2,
            height="70vh",
            object_fit="contain"
        )

        current_session_id = gr.State(value=None)

        # ✅ Styled & grouped status display
        with gr.Group(elem_id="status-group"):
            status_display = gr.HTML(
                "<div style='text-align:center; font-size: 28px; padding: 20px;'>No job currently started</div>"
            )


        def update_gallery_and_status(session_id):
            if not session_id:
                return [], gr.update(
                    value="<div style='text-align:center; font-size: 28px;'>No job currently started</div>")

            try:
                resp = requests.get(f"http://localhost:8000/sessions/{session_id}")
                if resp.ok:
                    session = resp.json()
                    status = session.get("status", "unknown").lower()

                    status_map = {
                        "pending": "⏳ <b>Pending...</b>",
                        "running": "🚀 <b>Running...</b>",
                        "completed": "✅ <b>Completed</b>",
                        "failed": "❌ <b>Failed</b>"
                    }
                    message = status_map.get(status, f"🔍 <b>Status:</b> {status}")

                    return load_best_architectures(session_id), gr.update(
                        value=f"<div style='text-align:center; font-size: 28px; padding: 20px;'>{message}</div>"
                    )
                else:
                    return [], gr.update(
                        value="<div style='text-align:center; font-size: 28px;'>⚠️ Error fetching session</div>")
            except Exception as e:
                return [], gr.update(value=f"<div style='text-align:center; font-size: 28px;'>⚠️ Exception: {e}</div>")


        # Timer that triggers the update
        timer = gr.Timer(5)
        timer.tick(
            fn=update_gallery_and_status,
            inputs=current_session_id,
            outputs=[gallery, status_display]
        )

    with gr.Tab("All Training Sessions"):
        all_sessions = gr.Dropdown(label="Available Sessions", choices=load_sessions(), value=None)
        session_gallery = gr.Gallery(label="Session Results", columns=2, height="70vh", object_fit="contain")
        selected_session_id = gr.State(value=None)
        refresh_timer = gr.Timer(2)  # Update every 2 seconds

        def on_session_change(session):
            session_id = extract_session_id(session)
            return session_id, load_best_architectures(session_id)

        all_sessions.change(
            fn=on_session_change,
            inputs=all_sessions,
            outputs=[selected_session_id, session_gallery]
        )

        def update_sessions_dropdown():
            return gr.update(choices=load_sessions())

        refresh_timer.tick(
            fn=update_sessions_dropdown,
            inputs=[],
            outputs=all_sessions
        )

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
            response = requests.post("http://localhost:8000/sessions/", json=payload)
            if response.status_code in [200, 201]:
                session_id = response.json().get("training_id")
                return f"✅ Job queued: {session_id}", session_id
            else:
                return f"❌ Server responded with {response.status_code}: {response.text}", None
        except Exception as e:
            return f"❌ Failed to send training request: {e}", None


    start_button.click(
        fn=start_training,
        inputs=[dataset_dropdown, gates_checklist, discount_rate, learning_rate, max_length, autoencoder_path],
        outputs=[status_output, current_session_id]
    )

if __name__ == "__main__":
    app.launch(debug=True)
