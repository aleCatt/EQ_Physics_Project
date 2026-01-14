import os
# optimize cpu threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import h5py
import numpy as np
from tqdm import tqdm

# ================= CONFIGURATION =================
EVENTS_PATH = 'Instance_events_counts_10k.hdf5' 
NOISE_PATH = 'Instance_noise_1k.hdf5'
OUTPUT_FILE = 'seismic_dataset_balanced.npz'

SAMPLE_RATE = 100
WINDOW_LENGTH = 6000  # 60 seconds
N_EVENTS = 1000
N_NOISE = 1000
TOTAL_SAMPLES = N_EVENTS + N_NOISE
# =================================================

def ensure_channels_first(waveforms):
    """
    Ensures waveform is (3, N).
    Fixes (N, 3) if necessary.
    """
    shape = waveforms.shape
    # If shape is (N, 3) where N > 3, transpose it
    if shape[0] > 3 and shape[1] == 3:
        return waveforms.T
    # If shape is (3, N), keep it
    elif shape[0] == 3:
        return waveforms
    else:
        # Fallback for unexpected shapes, assume channel is the smallest dim
        if shape[0] < shape[1]:
            return waveforms
        else:
            return waveforms.T

def load_and_process_data():
    print(f"--- Starting Data Preprocessing (Robust Fix) ---")
    
    # 1. PRE-ALLOCATE MEMORY
    print("Allocating memory...")
    X_final = np.zeros((TOTAL_SAMPLES, 3, WINDOW_LENGTH), dtype=np.float32)
    y_arrivals = np.full((TOTAL_SAMPLES, 2), -1.0, dtype=np.float32)
    
    current_idx = 0

    # 2. Process Earthquake Events
    if not os.path.exists(EVENTS_PATH):
        print(f"ERROR: {EVENTS_PATH} not found.")
        return

    print(f"Loading Events from {EVENTS_PATH}...")
    with h5py.File(EVENTS_PATH, 'r') as f:
        event_keys = list(f['data'].keys())
        
        np.random.seed(42)
        if len(event_keys) > N_EVENTS:
            selected_keys = np.random.choice(event_keys, N_EVENTS, replace=False)
        else:
            selected_keys = event_keys
            
        for key in tqdm(selected_keys, desc="Events"):
            ds = f['data'][key]
            raw_wave = ds[:] # Read raw data
            
            # FIX: Smart Shape Handling
            waveforms = ensure_channels_first(raw_wave)
            
            # Now waveforms is (3, N). N might be 6000 or 12000.
            
            # Crop or Pad to (3, 6000)
            length = waveforms.shape[1]
            if length > WINDOW_LENGTH:
                # Crop (take the first 6000 samples)
                waveforms = waveforms[:, :WINDOW_LENGTH]
            elif length < WINDOW_LENGTH:
                # Pad
                padding = WINDOW_LENGTH - length
                waveforms = np.pad(waveforms, ((0,0), (0, padding)), 'constant')
            
            # Normalize
            max_val = np.max(np.abs(waveforms))
            if max_val > 0:
                waveforms = waveforms / max_val
            
            # Assign
            X_final[current_idx] = waveforms.astype(np.float32)
            
            # Metadata
            # Access attributes safely
            try:
                p = float(ds.attrs.get('trace_P_arrival_sample'))
            except: p = -1.0
            
            try:
                s = float(ds.attrs.get('trace_S_arrival_sample'))
            except: s = -1.0

            y_arrivals[current_idx] = [p, s]
            
            current_idx += 1

    # 3. Process Noise
    if os.path.exists(NOISE_PATH):
        print(f"Loading Noise from {NOISE_PATH}...")
        with h5py.File(NOISE_PATH, 'r') as f:
            noise_keys = list(f['data'].keys())
            selected_keys = noise_keys[:N_NOISE]
            
            for key in tqdm(selected_keys, desc="Noise"):
                if current_idx >= TOTAL_SAMPLES: break
                
                ds = f['data'][key]
                raw_wave = ds[:]
                
                # FIX: Smart Shape Handling
                waveforms = ensure_channels_first(raw_wave)
                
                length = waveforms.shape[1]
                if length > WINDOW_LENGTH:
                    waveforms = waveforms[:, :WINDOW_LENGTH]
                elif length < WINDOW_LENGTH:
                    padding = WINDOW_LENGTH - length
                    waveforms = np.pad(waveforms, ((0,0), (0, padding)), 'constant')
                    
                max_val = np.max(np.abs(waveforms))
                if max_val > 0:
                    waveforms = waveforms / max_val
                
                X_final[current_idx] = waveforms.astype(np.float32)
                # Arrivals remain -1.0
                
                current_idx += 1
    else:
        print(f"WARNING: Noise file {NOISE_PATH} not found. Using only events.")
        # Trim arrays if noise is missing
        X_final = X_final[:current_idx]
        y_arrivals = y_arrivals[:current_idx]

    # 4. Save
    print(f"Saving dataset shape {X_final.shape} to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE, 
        waveforms=X_final, 
        arrivals=y_arrivals
    )
    print("SUCCESS.")

if __name__ == "__main__":
    load_and_process_data()
