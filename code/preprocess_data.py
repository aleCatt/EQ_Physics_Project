import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= CONFIGURATION =================
METADATA_PATH = 'metadata_Instance_events_10k.csv' 
EVENTS_PATH = 'Instance_events_counts_10k.hdf5' 
NOISE_PATH = 'Instance_noise_1k.hdf5'

OUTPUT_FILE = 'seismic_dataset_balanced.npz'

WINDOW_LENGTH = 6000
N_EVENTS = 1000
N_NOISE = 1000
TOTAL_SAMPLES = N_EVENTS + N_NOISE
# =================================================

def ensure_channels_first(waveforms):
    if waveforms.shape[0] > 3 and waveforms.shape[1] == 3:
        return waveforms.T
    return waveforms

def load_and_process_data():
    print("--- Preprocessing with CSV Metadata ---")
    
    # 1. LOAD METADATA
    if not os.path.exists(METADATA_PATH):
        print(f"CRITICAL ERROR: Metadata CSV not found at {METADATA_PATH}")
        print("Please check the 'metadata' folder in your dataset.")
        return

    print(f"Loading Metadata from {METADATA_PATH}...")
    # Low_memory=False prevents warning on mixed types
    df = pd.read_csv(METADATA_PATH, low_memory=False)
    
    # Create a dictionary for super-fast lookup: 
    # Key: trace_name -> Value: {P_sample, S_sample}
    # We clean the column names just in case there are extra spaces
    df.columns = [c.strip() for c in df.columns]
    
    # Check if we have the right columns
    if 'trace_name' not in df.columns:
        print(f"Error: CSV is missing 'trace_name'. Columns found: {df.columns}")
        return

    print("Building lookup table...")
    # Convert relevant columns to a dictionary
    meta_lookup = df.set_index('trace_name')[['trace_P_arrival_sample', 'trace_S_arrival_sample']].to_dict('index')

    # 2. PRE-ALLOCATE MEMORY
    X_final = np.zeros((TOTAL_SAMPLES, 3, WINDOW_LENGTH), dtype=np.float32)
    y_arrivals = np.full((TOTAL_SAMPLES, 2), -1.0, dtype=np.float32)
    current_idx = 0

    # 3. PROCESS EVENTS
    print(f"Scanning {EVENTS_PATH}...")
    with h5py.File(EVENTS_PATH, 'r') as f:
        event_keys = list(f['data'].keys())
        
        # Check first key to ensure it matches CSV format
        print(f"DEBUG: Sample H5 Key: {event_keys[0]}")
        if event_keys[0] not in meta_lookup:
            print("WARNING: The keys in HDF5 do not match 'trace_name' in CSV.")
            print("Trying to strip whitespace or file extensions...")
        
        valid_count = 0
        
        for key in tqdm(event_keys, desc="Matching Events"):
            if valid_count >= N_EVENTS:
                break
            
            # LOOKUP LABELS IN CSV
            if key not in meta_lookup:
                continue
                
            row = meta_lookup[key]
            p_arr = float(row['trace_P_arrival_sample'])
            s_arr = float(row['trace_S_arrival_sample'])
            
            # FILTER: Ensure valid P and S inside the 60s window
            if (p_arr >= 50) and (s_arr > p_arr) and (s_arr < WINDOW_LENGTH - 50):
                
                # Load Data
                ds = f['data'][key]
                raw_wave = ds[:]
                waveforms = ensure_channels_first(raw_wave)
                
                # Crop/Pad
                if waveforms.shape[1] > WINDOW_LENGTH:
                    waveforms = waveforms[:, :WINDOW_LENGTH]
                elif waveforms.shape[1] < WINDOW_LENGTH:
                    padding = WINDOW_LENGTH - waveforms.shape[1]
                    waveforms = np.pad(waveforms, ((0,0), (0, padding)), 'constant')
                
                # Normalize
                max_val = np.max(np.abs(waveforms))
                if max_val > 0:
                    waveforms = waveforms / max_val
                
                X_final[current_idx] = waveforms.astype(np.float32)
                y_arrivals[current_idx] = [p_arr, s_arr]
                
                valid_count += 1
                current_idx += 1

    print(f"Found {valid_count} valid events via CSV.")
    
    # Handle shortage of events
    if valid_count < N_EVENTS:
        print(f"Reducing dataset size because only {valid_count} valid events were found.")
        # Trim the pre-allocated array logic requires careful handling
        # We will slice at the end

    # 4. PROCESS NOISE
    if os.path.exists(NOISE_PATH):
        print(f"Loading Noise from {NOISE_PATH}...")
        with h5py.File(NOISE_PATH, 'r') as f:
            noise_keys = list(f['data'].keys())
            limit = min(N_NOISE, len(noise_keys))
            
            for key in tqdm(noise_keys[:limit], desc="Noise"):
                if current_idx >= TOTAL_SAMPLES: break
                
                ds = f['data'][key]
                raw_wave = ds[:]
                waveforms = ensure_channels_first(raw_wave)
                
                if waveforms.shape[1] > WINDOW_LENGTH:
                    waveforms = waveforms[:, :WINDOW_LENGTH]
                elif waveforms.shape[1] < WINDOW_LENGTH:
                    padding = WINDOW_LENGTH - waveforms.shape[1]
                    waveforms = np.pad(waveforms, ((0,0), (0, padding)), 'constant')
                    
                max_val = np.max(np.abs(waveforms))
                if max_val > 0:
                    waveforms = waveforms / max_val
                
                X_final[current_idx] = waveforms.astype(np.float32)
                current_idx += 1
    
    # 5. FINAL TRIM AND SAVE
    # If we found fewer events than expected, we trim the zeros at the end
    X_final = X_final[:current_idx]
    y_arrivals = y_arrivals[:current_idx]
    
    print(f"Saving final shape {X_final.shape} to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE, 
        waveforms=X_final, 
        arrivals=y_arrivals
    )
    print("SUCCESS.")

if __name__ == "__main__":
    load_and_process_data()
