#!/usr/bin/env python3
"""
Play noise. Noise is generated continuously in small chunks.
To control the color, set any PSD scaling exponent β. The range [-2,2] has been tested.
You can sweep the color parameter continuously at a given period.
Uses colorednoise library for noise/motion generation.
Uses the pyloudnorm library for perceptual loudness calculation and normalization 
based on ITU-R BS.1770 standards. Normalization is done on each chunk.

Author: Dobromir Dotov 2025
Tester: Aurelia Sofia Dotov
"""
import argparse
import time as systime
import numpy as np
assert np  # avoid "imported but unused" message (W0611)
from scipy.io.wavfile import write
import pyloudnorm as pyln
import colorednoise as cn
import warnings


class Oscillator:
    def __init__(self, a=-2, b=2, period = 60*5):
        self.a = a
        self.b = b
        self.omega = np.multiply(2*np.pi,period**-1) # 1/seconds
        self.phi = np.random.uniform(0,1)*2*np.pi
        self.y = self.phi_to_y(self.phi)
        self.t0 = systime.time()
        
    def __call__(self,t1=None):
        if t1 is None:
            t1 = systime.time()
        dt = t1 - self.t0
        self.t0 = t1
        self.phi = self.phi + dt*self.omega*2
        if self.phi>2*np.pi or self.phi<0:
            self.omega = -1*self.omega
        # self.y = self.phi_to_y_sine(self.phi)
        self.y = self.phi_to_y(self.phi)
        return self.y
        
    def phi_to_y(self,phi):
        y = phi/2/np.pi*(self.b-self.a)+self.a
        return y

    def phi_to_y_sine(self,phi):
        y = (np.real(np.exp(1j*phi))+1)/2*(self.b-self.a)+self.a
        return y

def amplify_waveform(waveform,level=1):
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        amplified_waveform = level*waveform/max_val
    else:
        amplified_waveform = waveform
    return amplified_waveform

def int_or_str(text): # For argument parsing.
    try:
        return int(text)
    except ValueError:
        return text

class Noise_Generator:
    def __init__(self, noise_dim=(100,1)):
        self.loudness0 = 0
        self.noise = np.zeros(noise_dim)
        self.len = self.noise.shape[0]
        
    def __call__(self, slope=1, target_loudness0=-27):
        for j in range(self.noise.shape[1]):
            x = cn.powerlaw_psd_gaussian(slope, self.noise.shape[0])
            self.loudness0 = meter.integrated_loudness(x)
            target_loudness = target_loudness0
            x = pyln.normalize.loudness(x, self.loudness0, target_loudness)
            trend = np.linspace(np.mean(x[0:100])-np.mean(self.noise[-100:,j]), 0, 4000)
            x[0:4000] = x[0:4000] - trend
            x = np.tanh(x)
            self.noise[:,j] = x

class Callback:
    def __init__(self,blocksize=1024,channels=1):
        self.prev_i = 0
        self.blocksize = blocksize
        self.data = np.zeros((self.blocksize,channels))
        self.index = np.arange(self.prev_i,self.blocksize,1)
        self.num_chunks_generated = 0

    def __call__(self, outdata=None, frames=None, time=None, status=None):
        global sound_frames
        if status:
            print(status)
        noise = noise_gen.noise
        self.next_i = self.prev_i + self.blocksize
        self.index = np.arange(self.prev_i, self.next_i, 1) % noise_gen.len
        self.data = noise[self.index,:]
        self.prev_i = self.next_i
        self.num_chunks_generated += 1
        if outdata is not None:
            outdata[:] = self.data
            if args.save:
                sound_frames.append(outdata*1)

        if self.index[-1] + 1 >= noise_gen.len:
            noise_gen(args.noise_slope,args.target_loudness)


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--list_devices', action='store_true',help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    import sounddevice as sd
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument('--device', type=int_or_str, help='output device (numeric ID or substring) or -1 to save the output to file')
parser.add_argument('--channels', type=int, default=1,help='number of channels [default 1]')
parser.add_argument('--samplerate', type=float, default=22050., help='sampling rate [default 22050]')
parser.add_argument('--noise_slope', type=float, default=1., help='the log-log slope of the noise scaling law [default 1]')
parser.add_argument('--duration', type=float, default=60., help='trial duration in minutes [default 60]')
parser.add_argument('--sweep_slope', type=float, default=0., help='the period in minutes of sweeping through the range of slopes {-2:2} [default 0]')
parser.add_argument('--save', type=bool, default=False, help='to store the sound [default 0]')
parser.add_argument('--device_info', type=bool, default=False, help='to show info about the sound device [default 0]')
parser.add_argument('--debugging', type=bool, default=False, help='to show figures with raw waveforms [default 0]')
parser.add_argument('--noise_amp', type=float, default=1.0, help='[Not being used currently] how much to scale the signal amplitude [default 1]')
args = parser.parse_args(remaining)

if args.debugging is False:
    warnings.filterwarnings("ignore", message="Possible clipped samples in output")

args.latency = None
args.player_callback_blocksize = .2 # seconds
args.blocksize = int(args.samplerate*args.player_callback_blocksize)
args.blocks_to_prepare_in_advance = 5
args.target_loudness = -37.0

# Initialize the engines
# Measure the loudness of each chunk of data
meter = pyln.Meter(args.samplerate)
meter.block_size = args.player_callback_blocksize

# Parameter oscillation sweep
if args.sweep_slope>0:
    args.sweep_period = args.sweep_slope*60 # seconds
    args.sweep_slope = True
    osc = Oscillator(period = args.sweep_period)
    args.noise_slope = osc()
    if args.device == -1:
        osc.t0 = 0
    

# Noise generator engine
noise_gen = Noise_Generator((int(args.blocksize*args.blocks_to_prepare_in_advance),args.channels))
noise_gen(args.noise_slope,args.target_loudness)
noise = noise_gen.noise

# This inserts data in the sound stream
callback = Callback(blocksize=args.blocksize,channels=args.channels)

if args.save or args.device == -1:
    if args.sweep_slope:
        wave_file = 'auditory_noise_beta_sweep_period_%.2fminutes' % float(args.sweep_period/60) + '.wav'
    else:
        wave_file = 'auditory_noise_beta_%2.2f' % args.noise_slope + '.wav' # + '_' + systime.strftime("%Y%m%d-%H%M%S")

# Sound player/writer
if args.device == -1:
    import wave
    o = None
    args.recording_size = int(args.duration*60*args.samplerate)
    args.number_of_blocks_in_recording = int(args.recording_size / args.blocksize)
    w = wave.open(wave_file, 'wb')
    w.setnchannels(args.channels)
    w.setsampwidth(2)
    w.setframerate(args.samplerate)
else:
    import sounddevice as sd
    sd.default.latency = 'low'
    if args.device is None:
        o = sd.query_devices(device=sd.default.device[1])
    else:
        o = sd.query_devices(device=args.device)
    if args.save:
        sound_frames = []

if args.debugging:
    from matplotlib import mlab
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for j in range(noise.shape[1]):
        ax[0].plot(noise[:,j],'--')
        s,f = mlab.psd(noise[:,j], NFFT=2**13)
        ax[1].loglog(f,s,'--')
        ax[1].grid(True)
    plt.tight_layout()
    plt.show(block=True)

if args.device_info or args.debugging:
    print('\n\n== Output device ==\n',o,'\n')
    for j in range(noise.shape[1]):
        print(f"Integrated loudness, pre: {noise_gen.loudness0:.2f} LUFS")
        loudness = meter.integrated_loudness(noise[:,j])
        print(f"Integrated loudness, post: {loudness:.2f} LUFS")
        print(f"Integrated loudness, target: {args.target_loudness:.2f} LUFS")
    
# Main loop
start_time = systime.time()
run_loop = True
while run_loop:
    if o is not None:
        try:
            with sd.OutputStream(device=o['index'],
                           dtype='float32',
                           latency=args.latency,
                           channels=args.channels, 
                           samplerate=args.samplerate, 
                           blocksize=args.blocksize,
                           callback=callback):
    
                print('To stop, press CTRL+C or wait %.2f minutes.' % args.duration)
                while (systime.time() - start_time) < args.duration*60:
                    systime.sleep(args.player_callback_blocksize/2)
                    if args.sweep_slope:
                        args.noise_slope = osc()
                    
                run_loop = False
            
        except KeyboardInterrupt:
            parser.exit('')
        except Exception as e:
            parser.exit(type(e).__name__ + ': ' + str(e))
    else:
        while callback.num_chunks_generated <= args.number_of_blocks_in_recording:
            callback()
            if args.sweep_slope:
                args.noise_slope = osc(t1=callback.next_i/args.samplerate)
            data = (callback.data*(2**15 - 1)).astype(np.int16)
            w.writeframes(data.tobytes())

        run_loop = False


# Save as WAV file
if args.save:
    data = np.vstack(sound_frames)
    data = np.multiply(data,255**2/2).astype(np.int16)
    write(wave_file, int(args.samplerate), data)
