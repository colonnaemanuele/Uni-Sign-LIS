import os
import json
import pandas as pd

# Percorsi principali
BASE_PATH = 'LIS/Continuous'
VIDEO_BASE = os.path.join(BASE_PATH, 'cropped_videos')
POSE_BASE = os.path.join(BASE_PATH, 'pose_format')
XLSX_BASE = os.path.join(BASE_PATH, 'xlsx')
OUTPUT_JSON = os.path.join(BASE_PATH, 'LIS_Labels.json')
LOG_FILE = os.path.join(BASE_PATH, 'missing_files.log')

# Folders to exclude
EXCLUDE_FOLDERS = {"30_10_2023", "21_11_2022", "07_12_2022", "03_11_2022"}

# Lista finale e log
final_data = []
log_entries = []

# Elenca tutte le cartelle (es. 12_10_2023)
for date_folder in sorted(os.listdir(VIDEO_BASE)):
    if date_folder in EXCLUDE_FOLDERS:
        continue
    video_dir = os.path.join(VIDEO_BASE, date_folder)
    pose_dir = os.path.join(POSE_BASE, date_folder)
    xlsx_path = os.path.join(XLSX_BASE, f"{date_folder}.xlsx")

    if not os.path.isdir(video_dir):
        log_entries.append(f"[{date_folder}] Cartella video mancante: {video_dir}")
        continue

    if not os.path.isdir(pose_dir):
        log_entries.append(f"[{date_folder}] Cartella pose mancante: {pose_dir}")
        continue

    if not os.path.isfile(xlsx_path):
        log_entries.append(f"[{date_folder}] File Excel mancante: {xlsx_path}")
        continue

    try:
        df = pd.read_excel(xlsx_path, header=None)
    except Exception as e:
        log_entries.append(f"[{date_folder}] Errore nella lettura di {xlsx_path}: {e}")
        continue

    for idx, row in df.iterrows():
        text = str(row[2]).strip()
        video_filename = f"{idx}.mp4"
        pose_filename = f"{idx}.pkl"

        video_path = os.path.join(video_dir, video_filename)
        pose_path = os.path.join(pose_dir, pose_filename)

        video_rel = f"{date_folder}/{idx}.mp4"
        pose_rel = f"{date_folder}/{idx}.pkl"

        if not os.path.isfile(video_path):
            log_entries.append(f"[{date_folder}] Video mancante: {video_rel}")
            continue

        if not os.path.isfile(pose_path):
            log_entries.append(f"[{date_folder}] Pose mancante: {pose_rel}")
            continue

        final_data.append({
            "video": video_rel,
            "pose": pose_rel,
            "text": text
        })

# Salva il JSON
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

# Scrivi il log
if log_entries:
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_entries))
    print(f"⚠️ Alcuni file mancanti. Log salvato in: {LOG_FILE}")
else:
    print("✅ Tutti i file presenti.")

print(f"✅ Creato {OUTPUT_JSON} con {len(final_data)} esempi validi.")