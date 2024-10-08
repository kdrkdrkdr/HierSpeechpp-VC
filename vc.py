import logging
import os
import torch
import numpy as np
from scipy.io.wavfile import write
import torchaudio
import utils
from torch.nn import functional as F
from hierspeechpp_speechsynthesizer import SynthesizerTrn, Wav2vec2
from speechsr24k.speechsr import SynthesizerTrn as SpeechSR24
from torchaudio.transforms import MelSpectrogram
from time import time
from rmvpe import RMVPE

# Configuration
CONFIG = {
    'output_dir': 'samples',
    'ckpt': './pretrained/hierspeechpp_v1.1_ckpt.pth',
    'ckpt_sr24': './speechsr24k/G_340000.pth',
    'scale_norm': 'prompt',
    'output_sr': 16000,
    'noise_scale_vc': 0.333,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(1234)

# Logging setup
logging.getLogger('numba').setLevel(logging.WARNING)

class MelSpectrogramFixed(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MelSpectrogramFixed, self).__init__()
        self.torchaudio_backend = MelSpectrogram(**kwargs)

    def forward(self, x):
        outputs = torch.log(self.torchaudio_backend(x) + 0.001)
        return outputs[..., :-1]

def load_models():
    hps = utils.get_hparams_from_file(os.path.join(os.path.split(CONFIG['ckpt'])[0], 'config.json'))
    h_sr24 = utils.get_hparams_from_file(os.path.join(os.path.split(CONFIG['ckpt_sr24'])[0], 'config.json'))

    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).to(device)

    w2v = Wav2vec2().to(device)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    net_g.load_state_dict(torch.load(CONFIG['ckpt']))
    net_g.eval()

    speechsr = SpeechSR24(
        h_sr24.data.n_mel_channels,
        h_sr24.train.segment_size // h_sr24.data.hop_length,
        **h_sr24.model).to(device)
    utils.load_checkpoint(CONFIG['ckpt_sr24'], speechsr, None)
    speechsr.eval()

    rmvpe = RMVPE('pretrained/rmvpe.pt', is_half=False, device=device)

    return net_g, speechsr, mel_fn, w2v, rmvpe, hps

def get_f0(audio, rmvpe, hop_length=80):
    f0 = rmvpe.infer_from_audio(audio)
    target_length = int(len(audio) / hop_length)
    f0 = np.pad(f0, (0, max(0, target_length - len(f0))))
    f0 = f0[:target_length]
    return f0[np.newaxis, np.newaxis, :]

def load_and_preprocess_audio(audio_path, target_sr=16000):
    audio, sr = torchaudio.load(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr, resampling_method="kaiser_window")
    return audio

def voice_conversion(source_speech, target_speech, models):
    net_g, speechsr, mel_fn, w2v, rmvpe, _ = models

    # Load and preprocess audio
    source_audio = load_and_preprocess_audio(source_speech)
    target_audio = load_and_preprocess_audio(target_speech)

    # Pad source audio
    p = (source_audio.shape[-1] // 1280 + 1) * 1280 - source_audio.shape[-1]
    source_audio = F.pad(source_audio, (0, p), mode='constant')

    # Extract and process F0
    start = time()
    s_f0 = get_f0(source_audio, rmvpe)
    print('src-f0', time()-start)
    
    start = time()
    t_f0 = get_f0(target_audio, rmvpe)
    print('tgt-f0', time()-start)

    ii = s_f0 != 0
    j = t_f0 != 0
    s_f0[ii] = (s_f0[ii] - s_f0[ii].mean()) / s_f0[ii].std()
    s_f0[ii] = ((s_f0[ii] * t_f0[j].std()) + t_f0[j].mean()).clip(min=0)
    denorm_f0 = torch.log(torch.FloatTensor(s_f0+1).to(device))

    # Process source audio for w2v
    y_pad = F.pad(source_audio.unsqueeze(0), (40, 40), "reflect").squeeze(0)
    x_w2v = w2v(y_pad.unsqueeze(0).to(device))
    x_length = torch.LongTensor([x_w2v.size(1)]).to(device)

    # Process target audio for mel spectrogram
    p = (target_audio.shape[-1] // 1600 + 1) * 1600 - target_audio.shape[-1]
    target_audio = F.pad(target_audio, (0, p), mode='constant')
    trg_mel = mel_fn(target_audio.unsqueeze(0).to(device))
    if trg_mel.dim() == 2:
        trg_mel = trg_mel.unsqueeze(0)
    trg_length = torch.LongTensor([trg_mel.size(-1)]).to(device)
    trg_length2 = torch.cat([trg_length, trg_length], dim=0)

    # Voice conversion
    with torch.no_grad():
        t1 = time()
        converted_audio = net_g.voice_conversion_noise_control(x_w2v, x_length, trg_mel, trg_length2, denorm_f0, noise_scale=CONFIG['noise_scale_vc'])
        t2 = time()
        if CONFIG['output_sr'] == 24000:
            converted_audio = speechsr(converted_audio)
        t3 = time()
        converted_audio = converted_audio / torch.abs(converted_audio).max() * 32767.0 * torch.max(target_audio.abs())
        t4 = time()
        converted_audio = converted_audio.squeeze().cpu().numpy().astype('int16')
        t5 = time()
        
    print(f'voice_conversion_noise_control {t2-t1}\nspeechsr {t3-t2}\naudio_abs {t4-t3}\nint16 {t5-t4}')

    # Save output
    file_name_s = os.path.splitext(os.path.basename(source_speech))[0]
    file_name_t = os.path.splitext(os.path.basename(target_speech))[0]
    file_name2 = f"{file_name_s}_to_{file_name_t}.wav"
    output_file = os.path.join(CONFIG['output_dir'], file_name2)
    write(output_file, CONFIG['output_sr'], converted_audio)
    del converted_audio
    return output_file

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['CURL_CA_BUNDLE'] = ''

    models = load_models()

    source_speech = './samples/ref.wav'
    target_speech = './samples/ref.wav'
    voice_conversion(source_speech, target_speech, models)
    
    print("\nRunning multiple conversions:")
    for i in range(10):
        start = time()
        voice_conversion(source_speech, target_speech, models)
        print(f"Conversion {i+1}: {time()-start:.6f}s")
        print("-" * 40)