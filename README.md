# rainbow_noise_machine

## Description
This python program generates different colors of auditory noise continuously in small chunks. It can vary the PSD beta scaling exponent from chunk to chunk. In this way, it can gradually sweep through the parameter range without producing and immediately noticeable change.

Equalizing loudness is another special feature of this program. It does its best to equalize the loudness of the noise in each chunk. Most existing approaches to generate noise, including commercial apps, are naive in this respect. Just changing the PSD slope creates drastic changes in spectral energy and human perception of loudness of the produce noise. Normalizing by total energy does not work great either because human hearing has very different sensitivities to different frequencies. Here loudness is controlled chunk by chunk using the pyloudnorm library which implements ITU-R BS.1770. Still, loudness is not perfect especially in the non-stationary range β>1 (Brownian motion) where the produced sound will tend to be much quieter.

## Requirements
Besides typical python packages such as numpy, this requires colorednoise, pyloudnorm, and sounddevice.

## Usage
The following line will play pink noise in two channels for one hour using the default device that sounddevice can find on your computer:
python noise_machine.py --duration 60 --noise_slope 1 --channels 2

The following line will write a 1-hour long .wav file with varying color noise in one channel. No sound will be played, the output goes straight to disk. To make things more fun, the PSD beta parameter is being swept back and forth in the range [-2,2] with a period of 10 minutes:
python noise_machine.py --duration 60 --sweep_slope 10 --device -1

## Examples
If you want to hear examples or just play the noise in your background without dealing with python installation, go to https://dobri.dotov.com/noise where I've put a few long examples.

