# File: build_myanmar_dataset.py
import os
import pandas as pd
import numpy as np
import h5py
from obspy import read, Stream
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MyanmarDatasetBuilder:
    def __init__(self, base_path=r"D:\datasets\myanmar_eq"):
        self.base_path = base_path
        self.eq_path = os.path.join(base_path, "earthquakes")
        self.noise_real_path = os.path.join(base_path, "myanmar_noise_real")
        self.noise_smart_path = os.path.join(base_path, "myanmar_noise_smart")
        self.metadata_path = os.path.join(base_path, "metadata")
        
        self.output_h5 = os.path.join(base_path, "myanmar_full.hdf5")
        self.output_csv = os.path.join(base_path, "myanmar_full.csv")
        
    def load_metadata(self):
        """Load all metadata files"""
        print("Loading metadata files...")
        
        # Earthquake metadata
        eq_csv = os.path.join(self.metadata_path, "earthquakes.csv")
        if os.path.exists(eq_csv):
            df_eq = pd.read_csv(eq_csv)
            print(f"  Earthquakes: {len(df_eq)} events")
        else:
            raise FileNotFoundError(f"Missing earthquakes.csv at {eq_csv}")
        
        # Pre-event noise metadata
        noise_real_csv = os.path.join(self.metadata_path, "myanmar_real_noise_extracted.csv")
        if os.path.exists(noise_real_csv):
            df_noise_real = pd.read_csv(noise_real_csv)
            print(f"  Pre-event noise: {len(df_noise_real)} segments")
        else:
            df_noise_real = None
            print("  Warning: No pre-event noise metadata found")
        
        # Ambient noise metadata
        noise_smart_csv = os.path.join(self.metadata_path, "myanmar_smart_noise.csv")
        if os.path.exists(noise_smart_csv):
            df_noise_smart = pd.read_csv(noise_smart_csv)
            print(f"  Ambient noise: {len(df_noise_smart)} segments")
        else:
            # Try to create from files if CSV doesn't exist
            df_noise_smart = self.create_noise_smart_metadata()
        
        return df_eq, df_noise_real, df_noise_smart
    
    def create_noise_smart_metadata(self):
        """Create metadata for smart noise if CSV doesn't exist"""
        files = [f for f in os.listdir(self.noise_smart_path) if f.endswith('.mseed')]
        
        metadata = []
        for i, filename in enumerate(files):
            try:
                # Parse from filename: myanmar_noise_GE_BOAB_20230101_020000.mseed
                parts = filename.replace('.mseed', '').split('_')
                network = parts[2]
                station = parts[3]
                date_str = parts[4]  # YYYYMMDD
                time_str = parts[5]  # HHMMSS
                
                metadata.append({
                    'filename': filename,
                    'network': network,
                    'station': station,
                    'start_time': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}",
                    'earthquake_time': None,  # Not earthquake-related
                    'time_diff': None,
                    'sampling_rate': 100.0,  # Assuming standard
                    'noise_type': 'AMBIENT'
                })
            except:
                metadata.append({
                    'filename': filename,
                    'network': 'UNKNOWN',
                    'station': 'UNKNOWN',
                    'start_time': None,
                    'earthquake_time': None,
                    'time_diff': None,
                    'sampling_rate': 100.0,
                    'noise_type': 'AMBIENT'
                })
        
        df = pd.DataFrame(metadata)
        output_path = os.path.join(self.metadata_path, "myanmar_smart_noise.csv")
        df.to_csv(output_path, index=False)
        print(f"  Created ambient noise metadata: {len(df)} files")
        return df
    
    def process_waveform(self, file_path, target_sr=100, duration_sec=60):
        """Process waveform to standard format"""
        try:
            # Read the file
            st = read(file_path)
            
            # Merge traces if multiple
            if len(st) > 1:
                st.merge(method=1, fill_value='interpolate')
            
            # Get first trace
            tr = st[0]
            
            # Resample if needed
            if tr.stats.sampling_rate != target_sr:
                tr.resample(target_sr)
            
            # Get or create 3 channels
            # For single channel data, replicate to 3 channels
            data_1d = tr.data
            
            # Ensure we have enough samples
            target_samples = int(duration_sec * target_sr)
            if len(data_1d) > target_samples:
                data_1d = data_1d[:target_samples]
            elif len(data_1d) < target_samples:
                # Pad with zeros
                pad_len = target_samples - len(data_1d)
                data_1d = np.pad(data_1d, (0, pad_len), mode='constant')
            
            # Create 3-channel data (repeat same data on all channels)
            # In real implementation, you'd want to handle actual 3-channel data
            data_3ch = np.stack([data_1d, data_1d, data_1d])
            
            # Detrend and normalize
            for ch in range(3):
                data_3ch[ch] = data_3ch[ch] - np.mean(data_3ch[ch])
                std = np.std(data_3ch[ch])
                if std > 0:
                    data_3ch[ch] = data_3ch[ch] / std
            
            return data_3ch.astype(np.float32)
            
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
            return None
    
    def build_dataset(self):
        """Main function to build the dataset"""
        print("\n" + "="*60)
        print("BUILDING MYANMAR EQ+NOISE DATASET")
        print("="*60)
        
        # Load metadata
        df_eq, df_noise_real, df_noise_smart = self.load_metadata()
        
        # Create HDF5 file
        with h5py.File(self.output_h5, 'w') as h5f:
            metadata_rows = []
            
            # Process earthquakes
            print("\nProcessing earthquakes...")
            for idx, row in tqdm(df_eq.iterrows(), total=len(df_eq)):
                filename = row.get('filename', f"eq_{idx}.mseed")
                file_path = os.path.join(self.eq_path, filename)
                
                if not os.path.exists(file_path):
                    # Try with common pattern
                    alt_path = os.path.join(self.eq_path, f"eq_{row.get('time', '')}_{row.get('station', '')}.mseed")
                    if os.path.exists(alt_path):
                        file_path = alt_path
                    else:
                        continue
                
                # Process waveform
                waveform = self.process_waveform(file_path)
                if waveform is not None:
                    # Save to HDF5
                    trace_name = f"eq_{idx:04d}"
                    h5f.create_dataset(trace_name, data=waveform, compression='gzip')
                    
                    # Add metadata
                    metadata_rows.append({
                        'trace_name': trace_name,
                        'label_eq': 1,
                        'source': 'MYANMAR_EQ',
                        'station': row.get('station', ''),
                        'network': row.get('network', ''),
                        'magnitude': row.get('magnitude', np.nan),
                        'time': row.get('time', ''),
                        'latitude': row.get('latitude', np.nan),
                        'longitude': row.get('longitude', np.nan),
                        'filename': filename
                    })
            
            # Process pre-event noise
            print("\nProcessing pre-event noise...")
            if df_noise_real is not None:
                for idx, row in tqdm(df_noise_real.iterrows(), total=len(df_noise_real)):
                    filename = row.get('filename', f"noise_real_{idx}.mseed")
                    file_path = os.path.join(self.noise_real_path, filename)
                    
                    if os.path.exists(file_path):
                        waveform = self.process_waveform(file_path)
                        if waveform is not None:
                            trace_name = f"noise_pre_{idx:04d}"
                            h5f.create_dataset(trace_name, data=waveform, compression='gzip')
                            
                            metadata_rows.append({
                                'trace_name': trace_name,
                                'label_eq': 0,
                                'source': 'MYANMAR_NOISE_PRE',
                                'station': row.get('station', ''),
                                'network': row.get('network', ''),
                                'magnitude': np.nan,
                                'time': row.get('start_time', ''),
                                'latitude': np.nan,
                                'longitude': np.nan,
                                'filename': filename
                            })
            
            # Process ambient noise
            print("\nProcessing ambient noise...")
            if df_noise_smart is not None:
                for idx, row in tqdm(df_noise_smart.iterrows(), total=len(df_noise_smart)):
                    filename = row.get('filename', f"noise_amb_{idx}.mseed")
                    file_path = os.path.join(self.noise_smart_path, filename)
                    
                    if os.path.exists(file_path):
                        waveform = self.process_waveform(file_path)
                        if waveform is not None:
                            trace_name = f"noise_amb_{idx:04d}"
                            h5f.create_dataset(trace_name, data=waveform, compression='gzip')
                            
                            metadata_rows.append({
                                'trace_name': trace_name,
                                'label_eq': 0,
                                'source': 'MYANMAR_NOISE_AMB',
                                'station': row.get('station', ''),
                                'network': row.get('network', ''),
                                'magnitude': np.nan,
                                'time': row.get('start_time', ''),
                                'latitude': np.nan,
                                'longitude': np.nan,
                                'filename': filename
                            })
        
        # Save metadata CSV
        df_metadata = pd.DataFrame(metadata_rows)
        df_metadata.to_csv(self.output_csv, index=False)
        
        print("\n" + "="*60)
        print("DATASET BUILDING COMPLETE!")
        print("="*60)
        print(f"Total samples: {len(df_metadata)}")
        print(f"  Earthquakes: {len(df_metadata[df_metadata['label_eq'] == 1])}")
        print(f"  Noise: {len(df_metadata[df_metadata['label_eq'] == 0])}")
        print(f"    - Pre-event: {len(df_metadata[df_metadata['source'] == 'MYANMAR_NOISE_PRE'])}")
        print(f"    - Ambient: {len(df_metadata[df_metadata['source'] == 'MYANMAR_NOISE_AMB'])}")
        print(f"\nOutput files:")
        print(f"  HDF5: {self.output_h5}")
        print(f"  CSV: {self.output_csv}")
        print("\nReady for training!")
        
        return df_metadata

# Run the builder
if __name__ == "__main__":
    builder = MyanmarDatasetBuilder()
    df_metadata = builder.build_dataset()