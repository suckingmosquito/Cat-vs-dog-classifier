import os
from PIL import Image

dataset_path = r"C:\Users\Advan Aigen\Documents\Dataset\PetImages"
removed_count = 0

for category in ["Cat", "Dog"]:
    folder = os.path.join(dataset_path, category)
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            img = Image.open(filepath)
            img.verify()
        except Exception:
            print(f"Hapus file rusak: {filepath}")
            os.remove(filepath)
            removed_count += 1

print(f"\nSelesai! Total {removed_count} file rusak dihapus.")