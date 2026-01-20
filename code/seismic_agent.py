import torch
import torch.nn as nn
import numpy as np
import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import sys
import os

# --- 1. CONFIGURATION ---
CLIENT_NAME = "INGV"
NETWORK = "IV"

# List of stations to try (Central Italy)
# GIGS: Gran Sasso (Underground)
# AQU: L'Aquila
STATION_LIST = ["GIGS", "AQU"]

# Channels to try in order of preference
# HH = High Broadband (100Hz+), BH = Broadband (20-40Hz), EH = Short Period
CHANNEL_PRIORITY = ["HH*", "BH*", "EH*"]

MODEL_PATH = "best_unet_model.pth"
SAMPLE_RATE = 100.0
WINDOW_LEN = 6000 

# --- 2. MODEL ARCHITECTURE ---
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels=3, n_classes=2):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool1d(2)
        self.enc4 = ConvBlock(64, 128)
        self.pool4 = nn.MaxPool1d(2)
        self.center = ConvBlock(128, 256)
        self.up4 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256, 128)
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        self.final = nn.Conv1d(16, n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        c = self.center(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(c), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

# --- 3. ROBUST FETCHING LOGIC ---
def get_data_from_network():
    """Tries multiple stations and channels to find valid data."""
    client = Client(CLIENT_NAME)
    
    # LOOK BACK 5 MINUTES (300s) to allow data to propagate to the server
    t_now = UTCDateTime.now()
    t_end = t_now - 300 
    t_start = t_end - 60 # Get 60 seconds of data

    print(f"🕒 Requesting time window: {t_start.strftime('%H:%M:%S')} - {t_end.strftime('%H:%M:%S')} UTC")

    for station in STATION_LIST:
        for channel in CHANNEL_PRIORITY:
            print(f"   Trying {NETWORK}.{station}.. ({channel})...", end=" ")
            try:
                st = client.get_waveforms(NETWORK, station, "*", channel, t_start, t_end)
                if len(st) >= 3:
                    print("✅ Success!")
                    return st, station
                else:
                    print(f"❌ Incomplete data ({len(st)} traces).")
            except Exception:
                print("❌ No data.")
                continue
    
    return None, None

def preprocess_stream(st):
    """Cleans, resamples, and formats ObsPy stream for the U-Net."""
    st.merge(fill_value=0)
    
    # Resample to 100Hz if needed
    if st[0].stats.sampling_rate != SAMPLE_RATE:
        st.resample(SAMPLE_RATE)
    
    # Sort to ensure Z, N, E order if possible, or just consistent 3 channels
    st.sort()
    
    # Extract first 3 traces
    data_array = np.zeros((3, WINDOW_LEN), dtype=np.float32)
    
    for i, tr in enumerate(st[:3]):
        n_samples = min(len(tr.data), WINDOW_LEN)
        data_array[i, :n_samples] = tr.data[:n_samples]
        
    # Normalize
    max_val = np.max(np.abs(data_array))
    if max_val > 0:
        data_array /= max_val
        
    return data_array, st[0].stats.starttime

# --- 4. MAIN ---

def run_agent():
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet1D().to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Critical: {MODEL_PATH} not found.")
        sys.exit(1)
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Retrieve Data
    print(f"📡 Connecting to INGV Network...")
    st, active_station = get_data_from_network()
    
    if st is None:
        print("\n❌ Failed to acquire valid data from any station in the list.")
        print("   Possible reasons: Internet connection, high server latency, or maintenance.")
        return

    # Process
    input_data, start_time = preprocess_stream(st)
    input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(device)

    # Infer
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()[0] 

    p_probs = output[0]
    s_probs = output[1]
    p_max = np.max(p_probs)
    s_max = np.max(s_probs)
    
    print("\n" + "=" * 50)
    print(f"🔎 REPORT: Station {active_station}")
    print(f"📅 Time: {start_time}")
    print("-" * 50)
    print(f"Model Confidence:")
    print(f"   P-Wave: {p_max*100:.1f}%")
    print(f"   S-Wave: {s_max*100:.1f}%")
    
    threshold = 0.2
    
    if p_max > threshold:
        p_idx = np.argmax(p_probs)
        p_time = start_time + (p_idx / SAMPLE_RATE)
        print(f"\n🚨 SEISMIC EVENT DETECTED!")
        print(f"   🌊 P-Arrival: {p_time.strftime('%H:%M:%S.%f')[:-3]}")
    else:
        print("\n✅ Status: Background Noise (No events detected)")
    print("=" * 50)

if __name__ == "__main__":
    run_agent()
