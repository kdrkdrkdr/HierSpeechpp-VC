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
import pesto

# Configuration
CONFIG = {
    'output_dir': 'samples',
    'ckpt': './pretrained/hierspeechpp_v1.1_ckpt.pth',
    'ckpt_sr24': './speechsr24k/G_340000.pth',
    'scale_norm': 'prompt',
    'output_sr': 24000,
    'noise_scale_vc': 0.333,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(1234)

class MelSpectrogramFixed(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MelSpectrogramFixed, self).__init__()
        self.torchaudio_backend = MelSpectrogram(**kwargs)

    def forward(self, x):
        outputs = torch.log(self.torchaudio_backend(x) + 0.001)
        return outputs[..., :-1]


def model_load():
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
    ).cuda()

    w2v = Wav2vec2().cuda()

    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    net_g.load_state_dict(torch.load(CONFIG['ckpt']))
    net_g.eval()

    speechsr = SpeechSR24(h_sr24.data.n_mel_channels,
        h_sr24.train.segment_size // h_sr24.data.hop_length,
        **h_sr24.model).cuda()
    utils.load_checkpoint(CONFIG['ckpt_sr24'], speechsr, None)
    speechsr.eval()

    return net_g, speechsr, mel_fn, w2v, hps



def get_pesto_f0(audio, hop_length=80, device='cuda'):
    audio = audio.mean(dim=0)
    t, f0, c, a = pesto.predict(audio.to(device), 16000, num_chunks=1)
    f0 = f0.unsqueeze(0)[0].cpu().numpy()
    target_length = int(audio.shape[0] / hop_length)
    f0 = np.pad(f0, (0, max(0, target_length - len(f0))))
    f0 = f0[:target_length]
    return f0[np.newaxis, np.newaxis, :]


def voice_conversion(source_speech, target_speech):
    source_audio, sample_rate = torchaudio.load(source_speech)
    if sample_rate != 16000:
        source_audio = torchaudio.functional.resample(source_audio, sample_rate, 16000, resampling_method="kaiser_window")
    p = (source_audio.shape[-1] // 1280 + 1) * 1280 - source_audio.shape[-1]
    source_audio = torch.nn.functional.pad(source_audio, (0, p), mode='constant').data

    f0 = get_pesto_f0(source_audio)

    ii = f0 != 0
    f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std()

    y_pad = F.pad(source_audio, (40, 40), "reflect")
    x_w2v = w2v(y_pad.cuda())
    x_length = torch.LongTensor([x_w2v.size(2)]).to(device)

    target_audio, sample_rate = torchaudio.load(target_speech)
    target_audio = target_audio[:1,:]
    if sample_rate != 16000:
        target_audio = torchaudio.functional.resample(target_audio, sample_rate, 16000, resampling_method="kaiser_window") 

    t_f0 = get_pesto_f0(target_audio)
    j = t_f0 != 0

    f0[ii] = ((f0[ii] * t_f0[j].std()) + t_f0[j].mean()).clip(min=0)
    denorm_f0 = torch.log(torch.FloatTensor(f0+1).cuda())
    ori_prompt_len = target_audio.shape[-1]
    p = (ori_prompt_len // 1600 + 1) * 1600 - ori_prompt_len
    target_audio = torch.nn.functional.pad(target_audio, (0, p), mode='constant').data

    trg_mel = mel_fn(target_audio.cuda())

    trg_length = torch.LongTensor([trg_mel.size(2)]).to(device)
    trg_length2 = torch.cat([trg_length, trg_length], dim=0)

    with torch.no_grad():
        converted_audio = net_g.voice_conversion_noise_control(x_w2v, x_length, trg_mel, trg_length2, denorm_f0, noise_scale=CONFIG['noise_scale_vc'])
        converted_audio = speechsr(converted_audio)
        converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 32767.0 * torch.max(target_audio.abs())
        converted_audio = converted_audio.squeeze().cpu().numpy()

    file_name_s = os.path.splitext(os.path.basename(source_speech))[0]
    file_name_t = os.path.splitext(os.path.basename(target_speech))[0]
    file_name2 = f"{file_name_s}_to_{file_name_t}.wav"
    output_file = os.path.join(CONFIG['output_dir'], file_name2)
    write(output_file, 24000, converted_audio)

    del converted_audio
    return output_file


if __name__ == '__main__':
    import os, warnings
    warnings.filterwarnings('ignore')
    os.environ['CURL_CA_BUNDLE'] = ''
    
    models = model_load()
    net_g, speechsr, mel_fn, w2v, hps = models

    source_speech = './samples/ref.wav'
    target_speech = './samples/ref.wav'
    voice_conversion(source_speech, target_speech)

    for i in range(10):
        start = time()
        voice_conversion(source_speech, target_speech)
        print(time()-start)