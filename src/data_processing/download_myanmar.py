"""
MAIN SCRIPT: Download Myanmar earthquake and noise data
Save as: D:\datasets\myanmar_eq\download_myanmar.py
"""
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import pandas as pd
import os
import sys
import time

# ========== CONFIGURATION ==========
BASE_PATH = r"D:\datasets\myanmar_eq"

# Create main directories
os.makedirs(os.path.join(BASE_PATH, "earthquakes"), exist_ok=True)
os.makedirs(os.path.join(BASE_PATH, "noise"), exist_ok=True)
os.makedirs(os.path.join(BASE_PATH, "metadata"), exist_ok=True)

print(f"Data will be saved to: {BASE_PATH}\n")

def download_earthquakes_simple(num_events=100):
    """Simple function to download earthquake data"""
    client = Client("IRIS")
    
    # Myanmar region
    region = {
        "minlatitude": 9.0,
        "maxlatitude": 28.0,
        "minlongitude": 92.0,
        "maxlongitude": 101.0
    }
    
    print("Step 1: Finding Myanmar earthquakes...")
    
    # Get earthquake list
    catalog = client.get_events(
        starttime=UTCDateTime(2020, 1, 1),
        endtime=UTCDateTime(2023, 12, 31),
        **region,
        minmagnitude=4.0,  # M4.0+ earthquakes
        limit=num_events * 3
    )
    
    print(f"Found {len(catalog)} earthquakes in catalog")
    
    earthquake_data = []
    downloaded = 0
    
    print("\nStep 2: Downloading waveforms...")
    
    for i, event in enumerate(catalog[:num_events]):
        try:
            origin = event.preferred_origin() or event.origins[0]
            magnitude = event.preferred_magnitude() or event.magnitudes[0]
            
            print(f"  {i+1}. M{magnitude.mag:.1f} at {origin.time.date}")
            
            # Download from Myanmar stations
            st = client.get_waveforms(
                network="MM,MY,GE",  # Myanmar networks
                station="*",
                location="*",
                channel="BHZ",  # Vertical component only
                starttime=origin.time - 30,
                endtime=origin.time + 90,
                attach_response=True
            )
            
            if len(st) == 0:
                print("     No data available")
                continue
            
            # Save each trace
            for tr in st:
                filename = f"eq_{origin.time.strftime('%Y%m%d_%H%M%S')}_{tr.stats.station}.mseed"
                filepath = os.path.join(BASE_PATH, "earthquakes", filename)
                tr.write(filepath, format="MSEED")
                
                earthquake_data.append({
                    'id': f"eq_{downloaded:04d}",
                    'filename': filename,
                    'station': tr.stats.station,
                    'network': tr.stats.network,
                    'magnitude': magnitude.mag,
                    'time': origin.time.isoformat(),
                    'latitude': origin.latitude,
                    'longitude': origin.longitude
                })
            
            downloaded += 1
            print(f"     Downloaded {len(st)} traces")
            
            # Be nice to the server
            time.sleep(0.5)
            
        except Exception as e:
            print(f"     Error: {e}")
            continue
    
    # Save metadata
    if earthquake_data:
        df = pd.DataFrame(earthquake_data)
        csv_file = os.path.join(BASE_PATH, "metadata", "earthquakes.csv")
        df.to_csv(csv_file, index=False)
        print(f"\n✅ Saved {len(df)} earthquake traces to {BASE_PATH}")
        print(f"   CSV file: {csv_file}")
    
    return earthquake_data

def download_noise_simple(num_traces=100):
    """Simple function to download noise data"""
    client = Client("IRIS")
    
    print("\nStep 3: Downloading noise data...")
    
    # Myanmar stations to try
    stations = [
        ("MM", "BUMA"),
        ("MM", "MYAN"),
        ("GE", "MDY"),
        ("GE", "YGN")
    ]
    
    noise_data = []
    downloaded = 0
    
    # Download 1-hour noise segments
    for year in [2021, 2022, 2023]:
        for month in [1, 4, 7, 10]:  # Different seasons
            for network, station in stations:
                if downloaded >= num_traces:
                    break
                
                try:
                    # Nighttime for quiet noise
                    start = UTCDateTime(year, month, 15, 2, 0, 0)  # 2 AM
                    end = start + 3600  # 1 hour
                    
                    st = client.get_waveforms(
                        network=network,
                        station=station,
                        location="*",
                        channel="BHZ",
                        starttime=start,
                        endtime=end
                    )
                    
                    if len(st) == 0:
                        continue
                    
                    tr = st[0]
                    filename = f"noise_{network}_{station}_{start.strftime('%Y%m%d_%H%M')}.mseed"
                    filepath = os.path.join(BASE_PATH, "noise", filename)
                    tr.write(filepath, format="MSEED")
                    
                    noise_data.append({
                        'id': f"noise_{downloaded:04d}",
                        'filename': filename,
                        'station': station,
                        'network': network,
                        'time': start.isoformat(),
                        'duration': 3600
                    })
                    
                    downloaded += 1
                    print(f"  Downloaded noise {downloaded}/{num_traces}: {network}.{station}")
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    continue
    
    # Save metadata
    if noise_data:
        df = pd.DataFrame(noise_data)
        csv_file = os.path.join(BASE_PATH, "metadata", "noise.csv")
        df.to_csv(csv_file, index=False)
        print(f"\n✅ Saved {len(df)} noise traces to {BASE_PATH}")
        print(f"   CSV file: {csv_file}")
    
    return noise_data

def main():
    """Main function"""
    print("=" * 60)
    print("MYANMAR SEISMIC DATA DOWNLOADER")
    print("=" * 60)
    
    try:
        # Download 100 earthquakes and 100 noise traces (small test)
        earthquakes = download_earthquakes_simple(num_events=100)
        noise = download_noise_simple(num_traces=100)
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Earthquake files: {len(earthquakes)}")
        print(f"Noise files: {len(noise)}")
        print(f"\nData location: {BASE_PATH}")
        print("\nFolders created:")
        print(f"  {BASE_PATH}\\earthquakes\\   - Earthquake waveforms (.mseed)")
        print(f"  {BASE_PATH}\\noise\\         - Noise waveforms (.mseed)")
        print(f"  {BASE_PATH}\\metadata\\      - CSV files with information")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()