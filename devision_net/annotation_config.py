import json

old_json = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\devision_net\devnet_data_new\all_dataset\annotations.json"
new_json = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\devision_net\devnet_data_new\all_dataset\annotations.json"

with open(old_json, "r", encoding="utf-8") as f:
    old_data = json.load(f)

with open(new_json, "r", encoding="utf-8") as f:
    new_data = json.load(f)

print("=== OLD annotations sample ===")
print(old_data["annotations"][0])

print("\n=== NEW annotations sample ===")
print(new_data["annotations"][0])
