# ui.py
import gradio as gr
import torch
from training import preprocess_audio, load_best_model

def predict_audio(audio_path):
    model = load_best_model()
    features = preprocess_audio(audio_path)
    input_tensor = torch.tensor(features).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1).squeeze().numpy()
    decision = "Real" if prob[0] > prob[1] else "Fake"
    return f"{decision} (Real: {prob[0]:.2f}, Fake: {prob[1]:.2f})"

iface = gr.Interface(
    fn=predict_audio,
    inputs=gr.Audio(type="filepath", label="Upload .wav audio"),
    outputs=gr.Text(label="Prediction"),
    title="Deepfake Audio Detection",
    description="Upload a 3-second .wav file to classify it as Real or Fake."
)
