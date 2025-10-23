# gui_recorder.py — GUI recorder + baby cry prediction
import tkinter as tk
from tkinter import messagebox, ttk
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import requests
import threading
import time

BACKEND_URL = "http://127.0.0.1:8000/predict"  # FastAPI backend
SR = 22050  # sample rate

root = tk.Tk()
root.title("Baby Cry Detection App")
root.geometry("600x650")
root.configure(bg="#f5d6eb")

canvas = tk.Canvas(root, width=600, height=650)
canvas.pack(fill="both", expand=True)
gradient = canvas.create_rectangle(0, 0, 600, 650, fill="#cce7ff", outline="")
canvas.lower(gradient)

header = tk.Label(root, text="👶 Baby Cry Detection App", font=("Arial", 20, "bold"), bg="#ec6ca8", fg="white")
header.place(x=50, y=10, width=500, height=50)

live_status_var = tk.StringVar(value="Monitoring your baby's sound in real-time...")
live_status = tk.Label(root, textvariable=live_status_var, font=("Arial", 14), bg="white", relief="ridge", bd=2)
live_status.place(x=50, y=80, width=500, height=50)

playback_label = tk.Label(root, text="Recorded audio will appear here after recording.", bg="white", relief="sunken")
playback_label.place(x=50, y=270, width=500, height=30)

start_btn = tk.Button(root, text="🎙 Start Recording", bg="#007bff", fg="white", font=("Arial", 12))
stop_btn = tk.Button(root, text="⏹ Stop Recording", bg="#dc3545", fg="white", font=("Arial", 12), state="disabled")
delete_btn = tk.Button(root, text="🗑 Delete Recording", bg="#6c757d", fg="white", font=("Arial", 12), state="disabled")

start_btn.place(x=50, y=200, width=150, height=50)
stop_btn.place(x=220, y=200, width=150, height=50)
delete_btn.place(x=390, y=200, width=150, height=50)

recording = False
audio_data = []

def update_live_status():
    while True:
        if not recording:
            # Randomly simulate cry detection
            status = "🚨 Baby cry detected!" if np.random.rand() < 0.1 else "Monitoring your baby's sound in real-time..."
            live_status_var.set(status)
        time.sleep(3)

def start_recording():
    global recording, audio_data
    audio_data = []
    recording = True
    start_btn.config(state="disabled")
    stop_btn.config(state="normal")
    delete_btn.config(state="disabled")

    def callback(indata, frames, time, status):
        audio_data.append(indata.copy())

    stream = sd.InputStream(callback=callback, channels=1, samplerate=SR)
    stream.start()
    start_recording.stream = stream

def stop_recording():
    global recording
    recording = False
    start_btn.config(state="normal")
    stop_btn.config(state="disabled")
    delete_btn.config(state="normal")
    start_recording.stream.stop()
    start_recording.stream.close()
    playback_label.config(text="Recording completed. Sending to backend...")

    # Save to temp WAV and send to backend in a thread
    threading.Thread(target=send_to_backend).start()

def delete_recording():
    global audio_data
    audio_data = []
    playback_label.config(text="Recording deleted.")
    delete_btn.config(state="disabled")

def send_to_backend():
    global audio_data
    audio_array = np.concatenate(audio_data, axis=0)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavfile.write(temp_file.name, SR, audio_array)
    
    try:
        with open(temp_file.name, "rb") as f:
            files = {"file": ("cry.wav", f, "audio/wav")}
            response = requests.post(BACKEND_URL, files=files)
        data = response.json()
        if data.get("error"):
            messagebox.showerror("Error", data["error"])
        else:
            result_text = f"Predicted reason: {data['top_reason']}\n\n"
            for reason, prob in data["probabilities"].items():
                result_text += f"{reason}: {prob:.2f}%\n"
            messagebox.showinfo("Prediction Result", result_text)
            playback_label.config(text="Prediction completed.")
    except Exception as e:
        messagebox.showerror("Error", f"Could not reach backend: {e}")
        playback_label.config(text="Prediction failed.")


start_btn.config(command=start_recording)
stop_btn.config(command=stop_recording)
delete_btn.config(command=delete_recording)

threading.Thread(target=update_live_status, daemon=True).start()

root.mainloop()
