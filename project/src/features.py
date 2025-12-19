"""
Feature extraction utilities (audio): MFCC and log-mel spectrogram extraction.
Generates fixed-size vectors by computing mean+std pooling over frames.
"""
import os
import argparse
import numpy as np
import librosa


def extract_mfcc_mean(path, sr=22050, n_mfcc=40, hop_length=512):
    y, sr = librosa.load(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    # Pool across time (mean and std)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feat


def extract_logmelspec(path, sr=22050, n_fft=2048, hop_length=512, n_mels=64, duration=None, out_frames=128):
    """Extract a log-mel spectrogram and return a fixed-size array (n_mels x out_frames).

    If the spectrogram has fewer frames than out_frames it will be padded with minimum value; if longer it will be truncated.
    """
    y, sr = librosa.load(path, sr=sr, duration=duration)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    logS = librosa.power_to_db(S, ref=np.max)
    # Normalize to zero mean, unit std per file
    logS = (logS - logS.mean()) / (logS.std() + 1e-9)

    # Frame handling
    if logS.shape[1] < out_frames:
        pad_width = out_frames - logS.shape[1]
        logS = np.pad(logS, ((0,0),(0,pad_width)), mode='constant', constant_values=logS.min())
    elif logS.shape[1] > out_frames:
        logS = logS[:, :out_frames]

    return logS.astype('float32')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/raw')
    parser.add_argument('--output_dir', type=str, default='data/features')
    parser.add_argument('--ext', type=str, default='.mp3')
    parser.add_argument('--mode', type=str, default='mfcc', choices=['mfcc', 'spec'], help='Feature mode: mfcc (pooled) or spec (log-mel spectrogram)')
    parser.add_argument('--n_mels', type=int, default=64)
    parser.add_argument('--out_frames', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for root, _, files in os.walk(args.input_dir):
        for fname in files:
            if not fname.lower().endswith(args.ext):
                continue
            path = os.path.join(root, fname)
            try:
                if args.mode == 'mfcc':
                    feat = extract_mfcc_mean(path)
                    outname = os.path.splitext(fname)[0] + '.npy'
                    np.save(os.path.join(args.output_dir, outname), feat)
                else:
                    spec = extract_logmelspec(path, n_mels=args.n_mels, out_frames=args.out_frames)
                    # Save spectrograms under a subdirectory 'spec'
                    spec_dir = os.path.join(args.output_dir, 'spec')
                    os.makedirs(spec_dir, exist_ok=True)
                    outname = os.path.splitext(fname)[0] + '.npy'
                    np.save(os.path.join(spec_dir, outname), spec)
            except Exception as e:
                print(f"Failed to process {path}: {e}")
