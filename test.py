from rmvpe import RMVPE
device = 'cuda'
model_path = 'pretrained/rmvpe.pt'
f0_extractor = RMVPE(model_path, is_half=False, device=device)


import torchaudio
def load_audio_16khz(audio):
    audio, sr = torchaudio.load(audio)
    audio = audio.mean(dim=0)
    if sr != 16000:
        source_audio = torchaudio.functional.resample(source_audio, sr, 16000, resampling_method="kaiser_window")
    return audio



a = f0_extractor.infer_from_audio(load_audio_16khz('samples/ref.wav'), thred=0.03)
print(a)
