import librosa
import numpy as np
import matplotlib.pyplot as plt

# Задаємо шлях до аудіофрагменту
audio_file = "ex2.mp3"

# Параметри фрагментації та швидкого перетворення Фур'є
frame_length = 2048
hop_length = 512

# Зчитуємо аудіодані з файлу
y, sr = librosa.load(audio_file, sr=None)

# Фрагментуємо аудіодані
fragments = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T

# До кожного з фрагментів застосовуємо швидке перетворення Фур'є
fft_fragments = np.fft.fft(fragments, axis=0)

# Приклад відображення амплітуди першого фрагменту
amplitude = np.abs(fft_fragments[:, 0])

# Приклад відображення фази першого фрагменту
phase = np.angle(fft_fragments[:, 0])

plt.figure(figsize=(12, 6))

# Графік амплітуди
plt.subplot(2, 1, 1)
plt.plot(amplitude)
plt.xlabel("Отсчеты")
plt.ylabel("Амплитуда")
plt.title("Амплитуда первого фрагмента")

# Графік фази
plt.subplot(2, 1, 2)
plt.plot(phase)
plt.xlabel("Отсчеты")
plt.ylabel("Фаза (радианы)")
plt.title("Фаза первого фрагмента")

plt.tight_layout()
plt.show()