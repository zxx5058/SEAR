import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks, butter, filtfilt, correlate, periodogram
from scipy.signal import butter, filtfilt
import glob
import re
from BlockVideo import split_video
from indexCaculate import calculate_mae, calculate_rmse, calculate_pcc, calculate_ccc, calculate_snr, calculate_mape, find_fundamental
import pandas as pd


def ImproveCrossingPoint(data, fs, shfit_distance, QualityLevel):
    N = len(data)
    data_shift = np.zeros(data.shape) - 1
    data_shift[shfit_distance:] = data[:-shfit_distance]
    cross_curve = data - data_shift

    zero_number = 0
    zero_index = []
    for i in range(len(cross_curve) - 1):
        if cross_curve[i] == 0:
            zero_number += 1
            zero_index.append(i)
        else:
            if cross_curve[i] * cross_curve[i + 1] < 0:
                zero_number += 1
                zero_index.append(i)

    cw = zero_number
    N = N
    fs = fs
    RR1 = ((cw / 2) / (N / fs)) * 60

    if (len(zero_index) <= 1 ) :
            RR2 = RR1
    else:
        time_span = 60 / RR1 / 2 * fs * QualityLevel
        zero_span = []
        for i in range(len(zero_index) - 1) :
            zero_span.append(zero_index[i + 1] - zero_index[i])

        while(min(zero_span) < time_span ) :
            doubt_point = np.argmin(zero_span)
            zero_index.pop(doubt_point)
            zero_index.pop(doubt_point)
            if len(zero_index) <= 1:
                break
            zero_span = []
            for i in range(len(zero_index) - 1):
                zero_span.append(zero_index[i + 1] - zero_index[i])

        zero_number = len(zero_index)
        cw = zero_number
        RR2 = ((cw / 2) / (N / fs)) * 60

    return RR2

# Detrend
def extract_brightness_signal(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    brightness_values = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        R = frame[:, :, 2]
        G = frame[:, :, 1]
        B = frame[:, :, 0]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        avg_brightness = np.mean(Y)
        brightness_values.append(avg_brightness)

    cap.release()
    brightness_values_detrended = np.array(brightness_values) - np.mean(brightness_values)
    brightness_values_detrended = highpass_filter(brightness_values_detrended, 0.1, frame_rate)
    return brightness_values_detrended, frame_rate


def extract_all_videos_brightness(video_folder):
    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]
    all_brightness_signals = []
    for video_path in video_files:
        print(f"Processing video: {video_path}")
        detrended_signal, frame_rate = extract_brightness_signal(video_path)
        all_brightness_signals.append(detrended_signal)
    return np.array(all_brightness_signals), frame_rate


def highpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, data)


def extract_timestamp_from_video_filename(video_filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})", video_filename)
    if match:
        timestamp = match.group(1)
        timestamp_clean = timestamp.replace("-", "").replace(" ", "").replace(":", "")
        return timestamp_clean


def plot_power_spectrum_with_peaks(signal, frame_rate):
    fft_values = np.fft.fft(signal)
    fft_frequencies = np.fft.fftfreq(len(signal), d=1 / frame_rate)

    positive_frequencies = fft_frequencies[:len(fft_frequencies) // 2]
    positive_fft_values = np.abs(fft_values[:len(fft_values) // 2])
    power_spectrum = np.square(positive_fft_values) / len(signal)
    peaks, _ = find_peaks(power_spectrum)

    peak_frequencies = positive_frequencies[peaks]
    peak_powers = power_spectrum[peaks]
    return peak_frequencies, peak_powers


def align_signals(signal_1, signal_2):
    correlation = correlate(signal_1, signal_2, mode='full')
    shift = np.argmax(np.abs(correlation)) - (len(signal_2) - 1)
    return shift


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    camera_folder = "./NormalData/Camera2"
    video_folder = "./output_blocks"
    npy_folder = "./NormalData/NPY"
    figure_files = './Results/Camera2'
    video_files = glob.glob(os.path.join(camera_folder, "*.mp4"))
    excel_file_path = './LTVCamera2.xlsx'

    total_mae = 0
    total_rmse = 0
    total_SNR = 0
    total_MAPE = 0
    all_RR_predict_values = []
    all_RR_real_values = []
    results = []
    T_values_list = []
    mae_list = []
    num_videos = len(video_files)
    for video_path in video_files:
        print(f"Processing video:{video_path}")
        split_video(video_path, video_folder)
        video_filename = os.path.basename(video_path)
        timestamp = extract_timestamp_from_video_filename(video_filename)
        npy_filename = timestamp + ".npy"
        npy_file_path = os.path.join(npy_folder, npy_filename)
        npydata = np.load(npy_file_path)

        npy_frame_count = npydata.shape[0]
        npy_frame_rate = 50
        npy_duration = npy_frame_count / npy_frame_rate

        # Extract signal from blocks
        all_brightness_signals, frame_rate = extract_all_videos_brightness(video_folder)
        video_frame_count = all_brightness_signals.shape[1]
        video_duration = video_frame_count / frame_rate
        if video_duration > npy_duration:
            all_brightness_signals = all_brightness_signals[:, :int(npy_duration * frame_rate)]
        elif video_duration < npy_duration:
            npydata = npydata[:int(video_duration * npy_frame_rate)]

        all_brightness_signals = StandardScaler().fit_transform(all_brightness_signals.T).T

        max_peak_frequencies = []
        max_peak_phases = []
        max_peak_powers = []
        second_max_peak_frequencies = []
        second_max_peak_powers = []
        valid_indices = []

        num_signals = all_brightness_signals.shape[0]
        for i in range(num_signals):
            plt.figure(figsize=(12, 6))
            signal = all_brightness_signals[i]
            peak_frequencies, peak_powers = plot_power_spectrum_with_peaks(all_brightness_signals[i], frame_rate)

            max_power = np.max(peak_powers)
            max_freq = peak_frequencies[np.argmax(peak_powers)]

            if not (0.1 <= max_freq <= 1):
                continue
            valid_indices.append(i)

            max_peak_frequencies.append(max_freq)
            max_peak_powers.append(max_power)

            fft_values = np.fft.fft(signal)
            fft_freqs = np.fft.fftfreq(len(signal), d = 1 / npy_frame_rate)
            idx = np.argmin(np.abs(fft_freqs - max_freq))
            max_phase = np.angle(fft_values[idx])
            max_peak_phases.append(max_phase)

            sorted_indices = np.argsort(peak_powers)
            second_max_power = peak_powers[sorted_indices[-2]] if len(peak_powers) > 1 else None
            second_max_freq = peak_frequencies[sorted_indices[-2]] if len(peak_powers) > 1 else None

            second_max_peak_frequencies.append(second_max_freq)
            second_max_peak_powers.append(second_max_power)

        overall_max_power = np.max(max_peak_powers)
        overall_max_freq = max_peak_frequencies[np.argmax(max_peak_powers)]
        freq_band_width = 0.05

        freq_counts = np.sum(np.array(max_peak_frequencies) == overall_max_freq)
        T_value = freq_counts / num_signals
        T_values_list.append(T_value)

        if T_value > 0.5:
            freq_mask = np.abs(np.array(max_peak_frequencies) - overall_max_freq) > freq_band_width
            filtered_frequencies = np.array(max_peak_frequencies)[freq_mask]
            filtered_powers = np.array(max_peak_powers)[freq_mask]

            if len(filtered_powers) > 0:
                overall_max_power = np.max(filtered_powers)
                overall_max_freq = filtered_frequencies[np.argmax(filtered_powers)]
            else:
                overall_max_power = np.max(second_max_peak_powers)
                overall_max_freq = second_max_peak_frequencies[np.argmax(second_max_peak_powers)]

        ref_signal = all_brightness_signals[np.argmax(max_peak_powers)]
        ref_fft = np.fft.fft(ref_signal)
        ref_fft_freqs = np.fft.fftfreq(len(ref_signal), d=1 / npy_frame_rate)
        ref_idx = np.argmin(np.abs(ref_fft_freqs - overall_max_freq))
        ref_phase = np.angle(ref_fft[ref_idx])

        phase_diffs = []
        for phase in max_peak_phases:
            delta = phase - ref_phase
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
            phase_diffs.append(delta)

        # -----------------Recover Signal-----------------=
        num_frames = all_brightness_signals.shape[1]
        num_channels = all_brightness_signals.shape[0]
        duration = num_frames / frame_rate
        time = np.linspace(0, duration, num_frames)

        freq_start = 0.1
        freq_end = 1.0
        num_bases = 50
        frequencies = np.linspace(freq_start, freq_end, num_bases)
        D = np.vstack([np.sin(2 * np.pi * f * time) for f in frequencies]).T

        alpha = 0.01
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
        S = np.zeros((D.shape[1], num_channels))


        for idx, i in enumerate(valid_indices):
            lasso.fit(D, all_brightness_signals[i, :])
            S[:, i] = lasso.coef_

        tau = 0.1
        selected_channels = []
        for idx, i in enumerate(valid_indices):
            coef_norm = np.linalg.norm(S[:, i])
            phase_diff = phase_diffs[idx]
            if coef_norm > tau and np.abs(phase_diff)  <= np.pi/2:
                selected_channels.append(i)

        if len(selected_channels) == 0:
            print("Warning: No channel passed selection, using fallback.")
            fallback_idx = np.argmax([np.linalg.norm(S[:, i]) for i in valid_indices])
            selected_channels = [valid_indices[fallback_idx]]
            weights = [1.0]
        else:
            weights = np.linalg.norm(S[:, selected_channels], axis=0)

        recovered_signal = np.average(all_brightness_signals[selected_channels, :], axis=0, weights=weights)
        target_frequency = overall_max_freq
        freq_band_width = 0.15

        low_freq = target_frequency - freq_band_width
        high_freq = target_frequency + freq_band_width
        fft_values = np.fft.fft(recovered_signal)
        fft_frequencies = np.fft.fftfreq(len(recovered_signal), d=1 / frame_rate)

        filtered_fft_values = np.zeros_like(fft_values)
        for k in range(len(fft_frequencies)):
            if low_freq <= abs(fft_frequencies[k]) <= high_freq:
                filtered_fft_values[k] = fft_values[k]

        filtered_signal = np.fft.ifft(filtered_fft_values)
        filtered_signal = bandpass_filter(filtered_signal, 0.1, 0.6, frame_rate)

        npydata = bandpass_filter(npydata, 0.1, 0.6, npy_frame_rate)
        RR_real = ImproveCrossingPoint(npydata, npy_frame_rate, shfit_distance=10, QualityLevel=0.6)
        RR_pred = ImproveCrossingPoint(filtered_signal, frame_rate, shfit_distance=10, QualityLevel=0.6)
        print("RR_real:", RR_real)
        print("RR_pred:", RR_pred)
        all_RR_predict_values.append(RR_pred)
        all_RR_real_values.append(RR_real)

        RR_real_normalized = normalize(npydata)
        RR_pred_normalized = normalize(filtered_signal)
        RR_real_normalized = np.real(RR_real_normalized)
        RR_pred_normalized = np.real(RR_pred_normalized)
        shift = align_signals(RR_real_normalized, RR_pred_normalized)

        RR_pred_aligned = np.roll(RR_pred_normalized, shift)
        time_npydata = np.arange(len(RR_real_normalized)) / npy_frame_rate
        time_filtered_signal = np.arange(len(RR_pred_aligned)) / frame_rate

        save_path = os.path.join(figure_files, f"{video_filename}_RR_comparison.png")
        plt.figure(figsize=(12, 6))
        plt.plot(time_npydata, RR_real_normalized, label="RR_real (Normalized)", color='b')
        plt.plot(time_filtered_signal, RR_pred_aligned, label="RR_pred (Normalized, Aligned)", color='r',
                 linestyle='--')

        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Value")
        plt.title("Aligned RR_real and RR_pred on Time Axis")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

        MAE = abs(RR_pred - RR_real)
        RMSE = (RR_pred - RR_real) ** 2
        print("MAE:", MAE)
        print("RMSE:", RMSE)
        total_mae += MAE
        total_rmse += RMSE

        mae_list.append(MAE)

        fundamental_freq = find_fundamental(RR_real_normalized, npy_frame_rate)
        print("fundamental_freq", fundamental_freq)
        snr_value = calculate_snr(RR_pred_normalized, fundamental_freq, frame_rate)
        print("SNR:", snr_value)
        if np.isinf(snr_value):
            snr_value = 0
        total_SNR += snr_value
        print("SNR:", snr_value)

        MAPE = abs((RR_real - RR_pred) / RR_real) * 100
        total_MAPE += MAPE
        print("MAPE:", MAPE)

        results.append({
            'video_filename': video_filename,
            'RR_real': RR_real,
            'RR_pred': RR_pred,
            'SNR': snr_value,
            'MAE': MAE,
            'RMSE': RMSE,
            'MAPE': MAPE
        })

    average_SNR = total_SNR / num_videos
    average_MAPE = total_MAPE / num_videos
    overall_PCC = calculate_pcc(all_RR_predict_values, all_RR_real_values) if len(all_RR_predict_values) > 1 else None
    overall_CCC = calculate_ccc(all_RR_real_values, all_RR_predict_values) if len(all_RR_real_values) > 1 else None
    total_mae /= num_videos
    total_rmse = np.sqrt(total_rmse / num_videos)
    print(f"Total MAE: {total_mae:.4f}")
    print(f"Total RMSE: {total_rmse:.4f}")
    print(f"Total SNR: {average_SNR:.4f}")
    print(f"Total MAPE: {average_MAPE:.4f}")
    print(f"Total PCC:{overall_PCC:.4f}")
    print(f"Total CCC:{overall_CCC:.4f}")

    df = pd.DataFrame(results)
    df.to_excel(excel_file_path, index=False)
    print(f"Results saved to {excel_file_path}")

    df_t_mae = pd.DataFrame({
        "T_value":T_values_list,
        "MAE":mae_list
    })
    df_t_mae.to_excel("./CMT_MAE_camera2.xlsx", index=False)




