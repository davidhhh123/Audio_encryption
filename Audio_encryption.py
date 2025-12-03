
import numpy as np


import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.linalg import hadamard
from scipy.fftpack import dct, idct

class AudioEncryptionWHTHenonWithSubstitution:
    def __init__(self, frame_length=1024, henon_a=3.58, henon_b=0.56, discard_iters=100, logistic_r=3.99):

        self.frame_length = frame_length
        self.henon_a = henon_a
        self.henon_b = henon_b
        self.discard_iters = discard_iters
        self.logistic_r = logistic_r


        if self.frame_length <= 0 or (self.frame_length & (self.frame_length - 1)) != 0:
            raise ValueError("frame_length должен быть степенью двойки и положительным.")
        self.H = hadamard(self.frame_length)

    def normalize_audio(self, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def split_audio_to_frames(self, audio_1d):
        total_len = len(audio_1d)
        num_frames = int(np.ceil(total_len / self.frame_length))
        padded_len = num_frames * self.frame_length
        padded_audio = np.pad(audio_1d, (0, padded_len - total_len), mode='constant')
        frames = padded_audio.reshape(num_frames, self.frame_length)
        return frames, total_len

    def frames_to_audio(self, frames, original_length):
        audio_1d = frames.flatten()[:original_length]
        return audio_1d

    def apply_wht_to_frames(self, frames):

        return np.dot(frames, self.H)

    def apply_iwht_to_frames(self, frames_wht):

        size = self.frame_length
        return np.dot(frames_wht, self.H) / size
    def apply_dct_2d(self, matrix):

        return dct(dct(matrix.T, norm='ortho').T, norm='ortho')

    def apply_idct_2d(self, dct_coeffs):

        return idct(idct(dct_coeffs.T, norm='ortho').T, norm='ortho')

    def generate_henon_permutation(self, frame):

        N = self.frame_length

        frm = frame.astype(float)
        mn = np.min(frm)
        mx = np.max(frm)
        if mx - mn > 1e-10:
            frm_norm = (frm - mn) / (mx - mn)
        else:
            frm_norm = np.zeros_like(frm)

        key1 = np.mean(frm_norm[0::2])
        key2 = np.mean(frm_norm[1::2])
        x = key1 - np.floor(key1)
        y = key2 - np.floor(key2)

        for _ in range(self.discard_iters):
            x, y = 1 - self.henon_a * np.cos(x) - self.henon_b * y, -x

        seq = np.empty(N, dtype=float)
        for i in range(N):
            x, y = 1 - self.henon_a * np.cos(x) - self.henon_b * y, -x
            seq[i] = x
        perm = np.argsort(seq)
        return perm

    def invert_permutation(self, perm):
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        return inv

    def generate_logistic_keystream(self, frame, length):


        frm = frame.astype(float)
        mn = np.min(frm)
        mx = np.max(frm)
        if mx - mn > 1e-10:
            frm_norm = (frm - mn) / (mx - mn)
        else:
            frm_norm = np.zeros_like(frm)

        x = (np.mean(frm_norm) + 0.123456789) % 1.0
        seq = np.empty(length, dtype=float)

        for _ in range(100):
            x = self.logistic_r * x * (1 - x)

        for i in range(length):
            x = self.logistic_r * x * (1 - x)
            seq[i] = x

        seq = seq - 0.5
        return seq

    def permute_and_substitute(self, wht_frames):

        num_frames = wht_frames.shape[0]
        permuted = np.zeros_like(wht_frames)
        self.permutations = []
        self.keystreams = []
        for i in range(num_frames):
            frame = wht_frames[i]
            perm = self.generate_henon_permutation(frame)
            self.permutations.append(perm)
            p = frame[perm]

            ks = self.generate_logistic_keystream(frame, self.frame_length)
            self.keystreams.append(ks)

            substituted = p + ks
            permuted[i] = substituted
        return permuted

    def inverse_permute_and_substitute(self, frames_substituted):
        num_frames = frames_substituted.shape[0]
        restored = np.zeros_like(frames_substituted)
        for i in range(num_frames):
            substituted = frames_substituted[i]
            ks = self.keystreams[i]
            perm = self.permutations[i]
            inv = self.invert_permutation(perm)

            p = substituted - ks

            restored[i] = p[inv]
        return restored

    def encrypt_audio(self, audio_data):
        frames, original_length = self.split_audio_to_frames(audio_data)

        wht_frames = self.apply_wht_to_frames(frames)
        wht_frames = self.apply_dct_2d(wht_frames)

        perm_sub = self.permute_and_substitute(wht_frames)

        perm_sub = self.apply_idct_2d(perm_sub)
        encrypted_frames = self.apply_iwht_to_frames(perm_sub)
        encrypted_audio = self.frames_to_audio(encrypted_frames, original_length)
        return encrypted_audio, encrypted_frames.shape, original_length

    def decrypt_audio(self, encrypted_audio, shape, original_length):
        padded = np.pad(encrypted_audio, (0, shape[0]*shape[1] - len(encrypted_audio)), mode='constant')
        enc_frames = padded.reshape(shape)

        wht_frames = self.apply_wht_to_frames(enc_frames)
        wht_frames = self.apply_dct_2d(wht_frames)

        inv = self.inverse_permute_and_substitute(wht_frames)

        inv = self.apply_idct_2d(inv)
        dec_frames = self.apply_iwht_to_frames(inv)

        dec_audio = self.frames_to_audio(dec_frames, original_length)
        return dec_audio

    def analyze_encryption_quality(self, original, encrypted):
        results = {}
        min_len = min(len(original), len(encrypted))
        correlation = np.corrcoef(original[:min_len], encrypted[:min_len])[0, 1]
        results['correlation'] = correlation

        orig = original[:min_len]
        enc = encrypted[:min_len]
        orig_norm = ((orig - np.min(orig)) / (np.ptp(orig) + 1e-10) * 255).astype(int)
        enc_norm = ((enc - np.min(enc)) / (np.ptp(enc) + 1e-10) * 255).astype(int)

        orig_hist, _ = np.histogram(orig_norm, bins=256, range=(0, 255))
        enc_hist, _ = np.histogram(enc_norm, bins=256, range=(0, 255))

        results['original_entropy'] = entropy(orig_hist + 1e-10)
        results['encrypted_entropy'] = entropy(enc_hist + 1e-10)
        results['mse'] = np.mean((orig - enc) ** 2)
        return results

    def plot_comparison(self, original, encrypted, decrypted, sample_rate=22050):
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        time = np.linspace(0, len(original) / sample_rate, len(original))

        axes[0, 0].plot(time, original)
        axes[0, 0].set_title('Original Audio')
        axes[1, 0].plot(time, encrypted)
        axes[1, 0].set_title('Encrypted Audio')
        axes[2, 0].plot(time, decrypted)
        axes[2, 0].set_title('Decrypted Audio')

        axes[0, 1].hist(original, bins=100)
        axes[0, 1].set_title("Original Histogram")
        axes[1, 1].hist(encrypted, bins=100)
        axes[1, 1].set_title("Encrypted Histogram")
        axes[2, 1].hist(decrypted, bins=100)
        axes[2, 1].set_title("Decrypted Histogram")

        plt.tight_layout()
        plt.show()



def demo_audio_encryption_henon_substitution(input_path="input.wav"):
    encryptor = AudioEncryptionWHTHenonWithSubstitution(
        frame_length=1024,
        henon_a=3.58, henon_b=0.56,
        discard_iters=100,
        logistic_r=3.99
    )

    audio_signal, sr = sf.read(input_path)
    print(f"Loaded audio: {len(audio_signal)} samples at {sr} Hz")

    if audio_signal.ndim > 1:
        audio_signal = audio_signal[:, 0]

    encrypted_audio, shape, orig_length = encryptor.encrypt_audio(audio_signal)
    decrypted_audio = encryptor.decrypt_audio(encrypted_audio, shape, orig_length)


    encryptor.plot_comparison(audio_signal, encrypted_audio, decrypted_audio, sr)
    decrypted_audio = encryptor.normalize_audio(decrypted_audio)

    sf.write("original.wav", audio_signal, sr)
    sf.write("encrypted.wav", encrypted_audio, sr)
    sf.write("decrypted.wav", decrypted_audio, sr)
    print("✓ Audio files saved")


def demo_synthetic_signal_henon_substitution():
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio_signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    encryptor = AudioEncryptionWHTHenonWithSubstitution(frame_length=1024)
    encrypted_audio, shape, orig_length = encryptor.encrypt_audio(audio_signal)




    decrypted_audio = encryptor.decrypt_audio(encrypted_audio, shape, orig_length)
    analysis = encryptor.analyze_encryption_quality(audio_signal, encrypted_audio)
    print("Synthetic signal analysis:", analysis)
    encryptor.plot_comparison(audio_signal, encrypted_audio, decrypted_audio, sr)



import ipywidgets as widgets
from IPython.display import display, Audio
import numpy as np
import soundfile as sf


output = widgets.Output()
plot_output = widgets.Output()


btn_load_original = widgets.FileUpload(description="📁 Загрузить аудио", accept=".wav", multiple=False)
btn_encrypt = widgets.Button(description="🔐 Зашифровать", button_style="warning")
btn_load_encrypted = widgets.FileUpload(description="📁 Загрузить шифр", accept=".wav", multiple=False)
btn_decrypt = widgets.Button(description="🔓 Дешифровать", button_style="success")


encryptor = AudioEncryptionWHTHenonWithSubstitution(frame_length=1024)
original_audio = None
encrypted_audio = None
decrypted_audio = None
shape_global = None
orig_length_global = None
sample_rate_global = None


def on_load_original(change):
    global original_audio, sample_rate_global

    output.clear_output()
    plot_output.clear_output()

    file_info = next(iter(btn_load_original.value.values()))
    audio_bytes = file_info['content']

    with open("original.wav", "wb") as f:
        f.write(audio_bytes)

    original_audio, sample_rate_global = sf.read("original.wav")
    if original_audio.ndim > 1:
        original_audio = original_audio[:,0]

    with output:
        print("✔ Оригинальное аудио загружено!")
        display(Audio(original_audio, rate=sample_rate_global))

btn_load_original.observe(on_load_original, names='value')


def on_encrypt_clicked(b):
    global encrypted_audio, shape_global, orig_length_global

    output.clear_output()
    plot_output.clear_output()

    if original_audio is None:
        with output:
            print("⚠ Сначала загрузите аудио.")
        return

    encrypted_audio, shape_global, orig_length_global = encryptor.encrypt_audio(original_audio)
    sf.write("encrypted.wav", encrypted_audio, sample_rate_global)

    with output:
        print("✔ Аудио зашифровано!")
        display(Audio(encrypted_audio, rate=sample_rate_global))

btn_encrypt.on_click(on_encrypt_clicked)


def on_load_encrypted(change):
    global encrypted_audio

    output.clear_output()
    plot_output.clear_output()

    file_info = next(iter(btn_load_encrypted.value.values()))
    enc_bytes = file_info['content']

    with open("encrypted.wav", "wb") as f:
        f.write(enc_bytes)

    encrypted_audio, _ = sf.read("encrypted.wav")

    with output:
        print("✔ Зашифрованный WAV загружен!")
        display(Audio(encrypted_audio, rate=sample_rate_global))

btn_load_encrypted.observe(on_load_encrypted, names='value')


def on_decrypt_clicked(b):
    global decrypted_audio

    output.clear_output()
    plot_output.clear_output()

    if encrypted_audio is None:
        with output:
            print("⚠ Сначала загрузите или создайте encrypted.wav.")
        return

    decrypted_audio = encryptor.decrypt_audio(encrypted_audio, shape_global, orig_length_global)
    sf.write("decrypted.wav", decrypted_audio, sample_rate_global)

    with output:
        print("✔ Файл успешно дешифрован!")
        display(Audio(decrypted_audio, rate=sample_rate_global))

btn_decrypt.on_click(on_decrypt_clicked)


tab = widgets.Accordion(children=[
    widgets.VBox([btn_load_original, btn_encrypt, btn_load_encrypted, btn_decrypt])
])
tab.set_title(0, "Управление")

display(tab)
display(output)
display(plot_output)

