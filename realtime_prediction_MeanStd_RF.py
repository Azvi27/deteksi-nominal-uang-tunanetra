import serial
import time
import joblib
import pandas as pd
import colorsys
import os
import numpy as np
import pygame # Pustaka untuk memutar suara

# --- Path ke model, scaler, dan max_rgb yang disimpan ---
# PASTIKAN INI SESUAI DENGAN PATH TEMPAT ANDA MENYIMPAN MODEL TERBAIK
MODEL_DIR = "trained_model_FIXED_RF_MeanStd_CV_Tuned_6400rows_v2" 
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model_MeanStdHSV_CV_Tuned_6400rows_v2.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_MeanStdHSV_CV_Tuned_6400rows_v2.joblib") 
MAX_RGB_PATH = os.path.join(MODEL_DIR, "max_rgb_values_FIXED_CV_Tuned_6400rows_v2.joblib")

# --- Path ke direktori file MP3 ---
# Mengubah AUDIO_DIR agar mencari file MP3 di direktori yang sama dengan skrip
AUDIO_DIR = "." # "." merepresentasikan direktori saat ini

# --- Konfigurasi Serial ---
SERIAL_PORT = '/dev/ttyUSB0' # GANTI INI SESUAI PORT ARDUINO ANDA
BAUD_RATE = 9600
# Harus sama dengan NUM_SAMPLES_PER_PREDICTION_BATCH di Arduino
NUM_SAMPLES_EXPECTED_FOR_PREDICT = 100 

# --- Variabel Global ---
loaded_model = None
loaded_scaler = None
loaded_max_rgb = None
is_pygame_initialized = False

def initialize_audio():
    """Menginisialisasi pygame mixer."""
    global is_pygame_initialized
    try:
        pygame.mixer.init()
        is_pygame_initialized = True
        print("Pygame mixer berhasil diinisialisasi untuk audio.")
    except Exception as e:
        print(f"Error saat menginisialisasi pygame mixer: {e}")
        is_pygame_initialized = False

def play_nominal_sound(nominal_value):
    """Memutar file MP3 yang sesuai dengan nominal yang diprediksi."""
    if not is_pygame_initialized:
        print("Audio tidak diinisialisasi, tidak dapat memutar suara.")
        return

    try:
        sound_file_name = f"{str(nominal_value)}.mp3"
        sound_file_path = os.path.join(AUDIO_DIR, sound_file_name)

        if os.path.exists(sound_file_path):
            print(f"Memutar suara untuk nominal {nominal_value}: {sound_file_path}")
            pygame.mixer.music.load(sound_file_path)
            pygame.mixer.music.play()
        else:
            print(f"File suara tidak ditemukan untuk nominal {nominal_value}: {sound_file_path}")
            print(f"Skrip mencari di: {os.path.abspath(sound_file_path)}") # Tambahkan ini untuk debugging path
    except Exception as e:
        print(f"Error saat memutar suara untuk nominal {nominal_value}: {e}")


def load_prediction_components():
    global loaded_model, loaded_scaler, loaded_max_rgb
    try:
        if not os.path.exists(MODEL_DIR):
            print(f"Error: Direktori model '{MODEL_DIR}' tidak ditemukan.")
            return False
        if not os.path.exists(MODEL_PATH):
            print(f"Error: File model '{MODEL_PATH}' tidak ditemukan.")
            return False
        if not os.path.exists(SCALER_PATH):
            print(f"Error: File scaler '{SCALER_PATH}' tidak ditemukan.")
            return False
        if not os.path.exists(MAX_RGB_PATH):
            print(f"Error: File max RGB '{MAX_RGB_PATH}' tidak ditemukan.")
            return False
            
        loaded_model = joblib.load(MODEL_PATH)
        loaded_scaler = joblib.load(SCALER_PATH)
        loaded_max_rgb = joblib.load(MAX_RGB_PATH)
        print("Model, scaler, dan nilai max RGB berhasil dimuat.")
        return True
    except FileNotFoundError as e:
        print(f"Error: File tidak ditemukan. Pastikan path benar dan skrip pelatihan sudah dijalankan.")
        print(f"Detail: {e}")
        return False
    except Exception as e:
        print(f"Error saat memuat komponen: {e}")
        return False

def predict_nominal_from_features(features_array):
    if loaded_model is None:
        print("Error: Model belum dimuat.")
        return None
    try:
        feature_names = ['H_mean', 'S_mean', 'V_mean', 'H_std', 'S_std', 'V_std']
        input_df = pd.DataFrame(features_array.reshape(1, -1), columns=feature_names)
        
        prediction = loaded_model.predict(input_df)
        predicted_nominal = prediction[0] 
        return predicted_nominal 
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return None

def process_rgb_batch_for_prediction(rgb_samples_list):
    if not rgb_samples_list or loaded_max_rgb is None or loaded_scaler is None:
        print("Error: Daftar sampel RGB kosong atau komponen (max_rgb/scaler) belum dimuat.")
        return None

    h_values, s_values, v_values = [], [], []

    for r_val, g_val, b_val in rgb_samples_list:
        try:
            r_norm = r_val / loaded_max_rgb['max_r']
            g_norm = g_val / loaded_max_rgb['max_g']
            b_norm = b_val / loaded_max_rgb['max_b']
            
            r_norm = max(0, min(1, r_norm))
            g_norm = max(0, min(1, g_norm))
            b_norm = max(0, min(1, b_norm))
            
            h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
            h_values.append(h)
            s_values.append(s)
            v_values.append(v)
        except ZeroDivisionError:
            print(f"Error: ZeroDivisionError saat normalisasi untuk sampel ({r_val},{g_val},{b_val}). Periksa max_rgb_values.")
            continue
        except Exception as e:
            print(f"Error memproses sampel RGB ({r_val},{g_val},{b_val}): {e}")
            continue

    if not h_values or len(h_values) < 2:
        print("Error: Tidak cukup sampel HSV yang valid untuk menghitung mean/std.")
        return None

    h_mean, s_mean, v_mean = np.mean(h_values), np.mean(s_values), np.mean(v_values)
    h_std, s_std, v_std = np.std(h_values), np.std(s_values), np.std(v_values)
    
    features_unscaled = np.array([h_mean, s_mean, v_mean, h_std, s_std, v_std])
    features_scaled = loaded_scaler.transform(features_unscaled.reshape(1, -1))
    
    return features_scaled

def main_realtime_prediction_with_sound():
    if not load_prediction_components():
        return
    
    initialize_audio() 

    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Mencoba terhubung ke Arduino di port {SERIAL_PORT}...")
        time.sleep(2) 
        if ser.is_open:
            print(f"Berhasil terhubung ke Arduino di port {SERIAL_PORT}.")
            print("Menunggu sentuhan pada sensor untuk memulai prediksi...")
        else:
            print(f"Gagal membuka port serial {SERIAL_PORT}.")
            return

        current_batch_rgb = []
        collecting_batch = False

        while True:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='replace').strip()
                    
                    if line == "CMD:START_BATCH_PREDICT":
                        print("\nINFO: Menerima START_BATCH_PREDICT dari Arduino.")
                        current_batch_rgb = []
                        collecting_batch = True
                        print(f"INFO: Mengumpulkan {NUM_SAMPLES_EXPECTED_FOR_PREDICT} sampel RGB...")
                        continue

                    if collecting_batch:
                        if line == "CMD:END_BATCH_PREDICT":
                            print(f"INFO: Menerima END_BATCH_PREDICT. Total sampel: {len(current_batch_rgb)}")
                            collecting_batch = False
                            if len(current_batch_rgb) == NUM_SAMPLES_EXPECTED_FOR_PREDICT:
                                print("INFO: Memproses batch RGB untuk prediksi...")
                                scaled_features_array = process_rgb_batch_for_prediction(current_batch_rgb)
                                
                                if scaled_features_array is not None:
                                    print(f"INFO: Fitur yang diskalakan (H_m,S_m,V_m,H_s,S_s,V_s): {np.round(scaled_features_array, 4)}")
                                    hasil_prediksi = predict_nominal_from_features(scaled_features_array)
                                    
                                    if hasil_prediksi is not None:
                                        print(f">>> PREDIKSI NOMINAL: {hasil_prediksi} <<<")
                                        play_nominal_sound(hasil_prediksi) 
                                    else:
                                        print("--- Gagal melakukan prediksi ---")
                                else:
                                    print("--- Error: Gagal memproses batch RGB ---")
                            else:
                                print(f"--- Error: Jumlah sampel diterima ({len(current_batch_rgb)}) != diharapkan ({NUM_SAMPLES_EXPECTED_FOR_PREDICT}) ---")
                            print("\nINFO: Menunggu sentuhan berikutnya...")
                            continue
                        
                        if line and not line.startswith("INFO:") and not line.startswith("CMD:"):
                            parts = line.split(',')
                            if len(parts) == 3:
                                r_raw = int(parts[0])
                                g_raw = int(parts[1])
                                b_raw = int(parts[2])
                                current_batch_rgb.append((r_raw, g_raw, b_raw))
                            
                    elif line.startswith("INFO:"):
                        print(line) 

                except UnicodeDecodeError:
                    print("DEBUG SERIAL: Error decoding data.")
                except ValueError as e:
                    print(f"DEBUG SERIAL: Error konversi data RGB. Data: '{line}'. Error: {e}")
                except Exception as e:
                    print(f"DEBUG SERIAL: Error saat memproses data serial: {e}")
            
            time.sleep(0.01) 

    except serial.SerialException as e:
        print(f"Error Serial: Tidak dapat membuka port {SERIAL_PORT}. Detail: {e}")
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna.")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("Koneksi serial ditutup.")
        if is_pygame_initialized:
            pygame.mixer.quit()
            print("Pygame mixer ditutup.")

if __name__ == "__main__":
    main_realtime_prediction_with_sound()
