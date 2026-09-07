import urllib.request
import json
import os

def download_reference_data():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/sources', exist_ok=True)
    
    # 1. Download Indian Railways stations / trains summary from datameet/railways
    trains_url = "https://raw.githubusercontent.com/datameet/railways/master/trains.json"
    print("Downloading datameet trains.json...")
    try:
        req = urllib.request.Request(trains_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            with open("data/raw/datameet_trains.json", "wb") as f:
                f.write(content)
            print(f"Saved datameet_trains.json ({len(content)} bytes)")
    except Exception as e:
        print(f"Error downloading datameet trains: {e}")

    stations_url = "https://raw.githubusercontent.com/datameet/railways/master/stations.json"
    print("Downloading datameet stations.json...")
    try:
        req = urllib.request.Request(stations_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            with open("data/raw/datameet_stations.json", "wb") as f:
                f.write(content)
            print(f"Saved datameet_stations.json ({len(content)} bytes)")
    except Exception as e:
        print(f"Error downloading datameet stations: {e}")

    # 2. Download sample of Hugging Face synthetic railway maintenance dataset
    hf_csv_url = "https://huggingface.co/datasets/shambhuraje/Indian_Railway_maintance/resolve/main/indian_railway_predictive_maintenance_100k.csv"
    print("Downloading sample rows from Hugging Face Indian_Railway_maintance...")
    try:
        # Download first 500KB to have a solid reference sample of ~1500 rows
        req = urllib.request.Request(hf_csv_url, headers={'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-512000'})
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            with open("data/raw/hf_indian_railway_maintenance_sample.csv", "wb") as f:
                f.write(content)
            print(f"Saved hf_indian_railway_maintenance_sample.csv ({len(content)} bytes)")
    except Exception as e:
        print(f"Error downloading HF dataset sample: {e}")

    # 3. Store Mendeley Railway Track Surface Faults Metadata
    mendeley_meta = {
        "dataset_name": "Railway Track Surface Faults Dataset",
        "doi": "10.17632/8hxtgyyxrw.2",
        "publisher": "Mendeley Data / Data in Brief (Elsevier)",
        "license": "CC BY 4.0",
        "fault_categories": [
            {"fault_name": "Squats", "type": "Rolling Contact Fatigue", "severity_range": [1, 5], "typical_department": "Engineering"},
            {"fault_name": "Cracks", "type": "Structural Rail Defect", "severity_range": [1, 5], "typical_department": "Engineering"},
            {"fault_name": "Flakings", "type": "Surface Degradation", "severity_range": [1, 4], "typical_department": "Engineering"},
            {"fault_name": "Shellings", "type": "Subsurface Fatigue", "severity_range": [1, 5], "typical_department": "Engineering"},
            {"fault_name": "Spallings", "type": "Severe Metal Loss", "severity_range": [2, 5], "typical_department": "Engineering"},
            {"fault_name": "Grooves", "type": "Mechanical Wear", "severity_range": [1, 4], "typical_department": "Engineering"},
            {"fault_name": "Joints Defect", "type": "Fishplate/Glued Joint Defect", "severity_range": [1, 5], "typical_department": "Engineering"}
        ]
    }
    with open("data/sources/mendeley_track_faults_meta.json", "w", encoding="utf-8") as f:
        json.dump(mendeley_meta, f, indent=2)
    print("Saved Mendeley metadata")

if __name__ == '__main__':
    download_reference_data()
