import pandas as pd

# Load the angles CSV you just created
df = pd.read_csv("output_angles.csv")

# Rule-based labeling function
def label_risk(row):
    if row['shoulder_angle'] > 150 or (row['knee_angle'] < 100 and row['spine_angle'] > 80):
        return 1  # Risky
    else:
        return 0  # Not risky

# Apply the risk label to each row
df['risk_label'] = df.apply(label_risk, axis=1)

# Save to a new CSV
df.to_csv("labeled_angles.csv", index=False)
print("Labeled dataset saved to labeled_angles.csv")
