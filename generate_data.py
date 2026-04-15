import pandas as pd
import random

conditions_data = [
    ("Flu", ["fever", "cough", "headache", "body ache", "fatigue", "sore throat"], "Paracetamol", "nausea, dizziness"),
    ("Migraine", ["headache", "nausea", "light sensitivity", "throbbing pain", "blurred vision"], "Sumatriptan", "drowsiness, tingling sensation"),
    ("Cold", ["sneezing", "runny nose", "cough", "mild fever", "congestion"], "Cetirizine", "dry mouth, tiredness"),
    ("Asthma", ["shortness of breath", "wheezing", "chest tightness", "coughing"], "Albuterol", "tremors, fast heart rate"),
    ("Diabetes", ["frequent urination", "excessive thirst", "unexplained weight loss", "fatigue"], "Metformin", "stomach upset, metallic taste"),
    ("Hypertension", ["high blood pressure", "severe headache", "fatigue", "vision problems", "chest pain"], "Amlodipine", "swelling of ankles, headache"),
    ("Allergy", ["itchy eyes", "runny nose", "skin rash", "sneezing", "hives"], "Loratadine", "headache, stomach pain"),
    ("Gastroenteritis", ["diarrhea", "stomach cramps", "nausea", "vomiting", "low-grade fever"], "Loperamide", "constipation, dizziness"),
    ("Insomnia", ["difficulty falling asleep", "waking up during night", "fatigue", "irritability"], "Zolpidem", "daytime drowsiness, dizziness"),
    ("Acid Reflux", ["heartburn", "chest pain", "difficulty swallowing", "regurgitation of food"], "Omeprazole", "headache, abdominal pain")
]

data = []
for i in range(200):
    condition, symptoms_list, drug, side_effects = random.choice(conditions_data)
    # Pick 2-4 symptoms randomly
    sampled_symptoms = random.sample(symptoms_list, k=random.randint(2, min(4, len(symptoms_list))))
    data.append({
        "condition": condition,
        "symptoms": ", ".join(sampled_symptoms),
        "drug": drug,
        "side_effects": side_effects
    })

df = pd.DataFrame(data)
df.to_csv("data/drug_dataset.csv", index=False)
print("Dataset generated successfully with 200 rows.")
