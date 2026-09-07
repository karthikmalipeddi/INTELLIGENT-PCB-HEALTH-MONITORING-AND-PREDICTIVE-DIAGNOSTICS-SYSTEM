import sqlite3

conn = sqlite3.connect("pcb_diagnostics.db")
cursor = conn.cursor()

cursor.execute("SELECT test_id, img_optical_rgb, img_thermal_raw, img_fused_hotspots FROM test_runs")
rows = cursor.fetchall()

for row in rows:
    test_id, optical_blob, thermal_blob, fused_blob = row

    if optical_blob:
        with open(f"test_{test_id}_optical.jpg", "wb") as f:
            f.write(optical_blob)
    if thermal_blob:
        with open(f"test_{test_id}_thermal.jpg", "wb") as f:
            f.write(thermal_blob)
    if fused_blob:
        with open(f"test_{test_id}_fused.jpg", "wb") as f:
            f.write(fused_blob)

print(f"Extracted all images for {len(rows)} test runs to your folder!")
conn.close()
