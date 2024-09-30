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
import torchcrepe
from time import time

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


# pesto_f0_extraction 으로 바꿔야함.
def get_crepe_f0(audio, rate=16000, hop_length=80, chunk_size=80000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # CREPE expects audio in the range [-1, 1]
    if audio.max() > 1 or audio.min() < -1:
        audio = audio / np.abs(audio).max()
    
    audio = torch.from_numpy(audio).float().to(device)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    f0_list = [
        torchcrepe.predict(
            audio[:, i:i+chunk_size],
            rate,
            hop_length=hop_length,
            fmin=50,
            fmax=1100,
            model='full',
            return_periodicity=True,
            device=device
        )[0].squeeze().cpu().numpy()
        for i in range(0, audio.shape[1], chunk_size)
    ]

    # Concatenate all chunks
    f0 = np.concatenate(f0_list)
    
    # Adjust the shape to match the expected input
    target_length = int(audio.shape[1] / hop_length)
    f0 = np.pad(f0, (0, max(0, target_length - len(f0))))
    f0 = f0[:target_length]
    
    return f0[np.newaxis, np.newaxis, :]

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


def voice_conversion(source_speech, target_speech):
    start = time()
    source_audio, sample_rate = torchaudio.load(source_speech)
    if sample_rate != 16000:
        source_audio = torchaudio.functional.resample(source_audio, sample_rate, 16000, resampling_method="kaiser_window")
    p = (source_audio.shape[-1] // 1280 + 1) * 1280 - source_audio.shape[-1]
    source_audio = torch.nn.functional.pad(source_audio, (0, p), mode='constant').data
    file_name_s = os.path.splitext(os.path.basename(source_speech))[0]
    print('1 >', time()-start)
    
    start = time()
    f0 = get_crepe_f0(source_audio.squeeze().numpy())
    print('2 >', time()-start)
    ii = f0 != 0
    f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std()

    start = time()
    y_pad = F.pad(source_audio, (40, 40), "reflect")
    x_w2v = w2v(y_pad.cuda())
    print('3 >', time()-start)
    x_length = torch.LongTensor([x_w2v.size(2)]).to(device)

    # Prompt load
    start = time()
    target_audio, sample_rate = torchaudio.load(target_speech)
    target_audio = target_audio[:1,:]
    if sample_rate != 16000:
        target_audio = torchaudio.functional.resample(target_audio, sample_rate, 16000, resampling_method="kaiser_window") 
    prompt_audio_max = torch.max(target_audio.abs())
    print('4 >', time()-start)

    start = time()
    t_f0 = get_crepe_f0(target_audio.squeeze().numpy())
    j = t_f0 != 0
    print('5 >', time()-start)

    start = time()
    f0[ii] = ((f0[ii] * t_f0[j].std()) + t_f0[j].mean()).clip(min=0)
    denorm_f0 = torch.log(torch.FloatTensor(f0+1).cuda())
    ori_prompt_len = target_audio.shape[-1]
    p = (ori_prompt_len // 1600 + 1) * 1600 - ori_prompt_len
    target_audio = torch.nn.functional.pad(target_audio, (0, p), mode='constant').data
    print('6 >', time()-start)

    file_name_t = os.path.splitext(os.path.basename(target_speech))[0]

    start = time()
    trg_mel = mel_fn(target_audio.cuda())
    print('8 >', time()-start)

    trg_length = torch.LongTensor([trg_mel.size(2)]).to(device)
    trg_length2 = torch.cat([trg_length, trg_length], dim=0)

    with torch.no_grad():
        start = time()
        converted_audio = net_g.voice_conversion_noise_control(x_w2v, x_length, trg_mel, trg_length2, denorm_f0, noise_scale=CONFIG['noise_scale_vc'])
        print('9 >', time()-start)

        # start = time()
        # converted_audio = speechsr(converted_audio)
        # print('10 >', time()-start)

    start = time()
    converted_audio = converted_audio.squeeze().cpu().numpy()
    print('11 >', time()-start)

    file_name2 = f"{file_name_s}_to_{file_name_t}.wav"
    output_file = os.path.join(CONFIG['output_dir'], file_name2)
    write(output_file, 16000, converted_audio)

    del converted_audio
    return output_file


if __name__ == '__main__':
    import os, warnings
    warnings.filterwarnings('ignore')
    os.environ['CURL_CA_BUNDLE'] = ''
    
    models = model_load()
    net_g, speechsr, mel_fn, w2v, hps = models

    source_speech = './samples/src.wav'
    target_speech = './samples/ref.wav'
    voice_conversion(source_speech, target_speech)

    for i in range(10):
        start = time()
        voice_conversion(source_speech, target_speech)
        print(time()-start)