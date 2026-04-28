"""
generate_synthetic_data.py

Generates a synthetic student dataset for demonstrating the Student Success
Index methodology. All students, schools, and outcomes are entirely fictional.

The dataset simulates 8th grade students at Shermer Middle School transitioning
to Ridgemont High School in a fictional California district, with fabricated
academic records and randomly assigned 9th grade outcomes.

Output: synthetic_students.csv

Usage:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --seed 42 --n 200
"""

import argparse
import random
import numpy as np
import pandas as pd

# ── Command-line args ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--n", type=int, default=180, help="Number of synthetic students")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

N = args.n

# ── Synthetic name components (clearly fictional) ─────────────────────────────
first_names = [
    "Aaliyah","Adrian","Alejandro","Amara","Amelia","Andre","Angela","Anthony",
    "Brianna","Carlos","Carmen","Christian","Claudia","Daniel","David","Diana",
    "Diego","Elena","Emilio","Emma","Eric","Fatima","Felix","Gabriel","Gloria",
    "Grace","Hannah","Hector","Isabella","Jada","Jamal","Jasmine","Jason",
    "Jennifer","Jessica","Jorge","Jose","Juan","Julia","Julian","Karen","Kevin",
    "Kira","Lena","Leonardo","Lila","Luis","Luna","Marco","Maria","Mario",
    "Maya","Michael","Miguel","Monica","Nathan","Nicole","Omar","Oscar","Pablo",
    "Patricia","Pedro","Rafael","Ramon","Rebecca","Ricardo","Rosa","Ryan",
    "Samuel","Sandra","Santiago","Sara","Serena","Sofia","Stephanie","Steven",
    "Talia","Thomas","Valentina","Victor","Xiomara","Yasmin","Zoe"
]

last_names = [
    "Alvarez","Anderson","Bautista","Brown","Castro","Chen","Cruz","Davis",
    "Diaz","Fernandez","Flores","Garcia","Gomez","Gonzalez","Guzman","Harris",
    "Hernandez","Jackson","Johnson","Jones","Lee","Lewis","Lopez","Lozano",
    "Martinez","Mendez","Mendoza","Miller","Morales","Moreno","Nguyen","Ortiz",
    "Perez","Ramirez","Ramos","Reyes","Rivera","Robinson","Rodriguez","Romero",
    "Ruiz","Salazar","Sanchez","Santos","Silva","Smith","Suarez","Taylor",
    "Thomas","Torres","Vargas","Vasquez","Vega","Williams","Wilson","Zamora"
]

# ── Demographic distributions (calibrated to reflect equity gaps in real data) ─
race_ethnicity_choices = [
    "Hispanic/Latino", "White", "African American", "Asian",
    "Two or More Races", "Pacific Islander", "American Indian"
]
race_weights = [0.42, 0.28, 0.12, 0.08, 0.06, 0.02, 0.02]

gender_choices = ["Female", "Male", "Non-binary"]
gender_weights = [0.50, 0.48, 0.02]

ell_choices = ["Yes", "No"]
ell_weights = [0.30, 0.70]

econ_dis_choices = ["Yes", "No"]
econ_dis_weights = [0.55, 0.45]

sped_choices = ["Yes", "No"]
sped_weights = [0.12, 0.88]

# ── GPA generation with equity-aware correlations ────────────────────────────
def generate_gpa(race, ell, econ_dis, subject="overall"):
    """
    Generate a plausible 8th grade GPA with realistic equity gaps baked in.
    Non-controllable factors shift the distribution; individual variation is large.
    """
    base = np.random.normal(2.6, 0.7)

    # Equity gap adjustments (reflect real systemic patterns, not individual destiny)
    if race in ["Hispanic/Latino", "African American"]:
        base -= np.random.uniform(0.1, 0.3)
    if ell == "Yes":
        base -= np.random.uniform(0.05, 0.25)
        if subject in ["math", "science"]:
            base += np.random.uniform(0.0, 0.15)  # ELL students sometimes stronger in STEM
    if econ_dis == "Yes":
        base -= np.random.uniform(0.05, 0.2)

    # Subject-specific variation
    if subject == "math":
        base += np.random.normal(0, 0.3)
    elif subject == "science":
        base += np.random.normal(0, 0.35)
    elif subject == "ela":
        base += np.random.normal(0, 0.25)

    return round(np.clip(base, 0.0, 4.0), 2)

def generate_attendance(race, ell, econ_dis):
    base = np.random.normal(94.5, 4.5)
    if econ_dis == "Yes":
        base -= np.random.uniform(0.5, 2.0)
    if race in ["African American"]:
        base -= np.random.uniform(0.3, 1.5)
    return round(np.clip(base, 60.0, 100.0), 1)

def generate_outcome(overall_gpa, math_gpa, science_gpa, attendance,
                     race, ell, econ_dis, sped):
    """
    Assign 9th grade DFI outcome probabilistically based on 8th grade factors.
    Higher risk with lower GPA, lower attendance, and compounding disadvantages.
    Returns 1 if student received a D, F, or I in 9th grade; 0 otherwise.
    """
    # Base risk from academic performance
    risk = 0.15  # baseline ~15% DFI rate

    if overall_gpa < 2.0:
        risk += 0.35
    elif overall_gpa < 2.5:
        risk += 0.20
    elif overall_gpa < 3.0:
        risk += 0.08

    if math_gpa < 1.5:
        risk += 0.20
    elif math_gpa < 2.0:
        risk += 0.10

    if science_gpa < 1.5:
        risk += 0.18
    elif science_gpa < 2.0:
        risk += 0.09

    if attendance < 90:
        risk += 0.20
    elif attendance < 95:
        risk += 0.08

    # Systemic risk factors (included in algorithm, not shown to students)
    if race in ["Hispanic/Latino", "African American"]:
        risk += 0.06
    if ell == "Yes":
        risk += 0.04
    if econ_dis == "Yes":
        risk += 0.05
    if sped == "Yes":
        risk += 0.08

    risk = np.clip(risk, 0.02, 0.92)
    return int(np.random.random() < risk)

# ── Generate student records ──────────────────────────────────────────────────
records = []
used_names = set()

for i in range(N):
    # Generate unique name
    for _ in range(100):
        first = random.choice(first_names)
        last = random.choice(last_names)
        name = f"{first} {last}"
        if name not in used_names:
            used_names.add(name)
            break

    student_id = f"SMS{str(i + 1001).zfill(4)}"  # Shermer Middle School ID

    race = np.random.choice(race_ethnicity_choices, p=race_weights)
    gender = np.random.choice(gender_choices, p=gender_weights)
    ell = np.random.choice(ell_choices, p=ell_weights)
    econ_dis = np.random.choice(econ_dis_choices, p=econ_dis_weights)
    sped = np.random.choice(sped_choices, p=sped_weights)

    overall_gpa = generate_gpa(race, ell, econ_dis, "overall")
    math_gpa = generate_gpa(race, ell, econ_dis, "math")
    science_gpa = generate_gpa(race, ell, econ_dis, "science")
    ela_gpa = generate_gpa(race, ell, econ_dis, "ela")
    attendance = generate_attendance(race, ell, econ_dis)

    outcome_dfi = generate_outcome(
        overall_gpa, math_gpa, science_gpa, attendance,
        race, ell, econ_dis, sped
    )

    records.append({
        "student_id": student_id,
        "last_name": last,
        "first_name": first,
        "gender": gender,
        "race_ethnicity": race,
        "english_learner": ell,
        "economically_disadvantaged": econ_dis,
        "special_education": sped,
        "grade8_overall_gpa": overall_gpa,
        "grade8_math_gpa": math_gpa,
        "grade8_science_gpa": science_gpa,
        "grade8_ela_gpa": ela_gpa,
        "grade8_attendance_pct": attendance,
        "grade9_any_dfi": outcome_dfi,
        "sending_school": "Shermer Middle School",
        "receiving_school": "Ridgemont High School",
    })

df = pd.DataFrame(records)

print(f"Generated {N} synthetic students")
print(f"DFI rate: {df['grade9_any_dfi'].mean()*100:.1f}%")
print(f"\nRace/ethnicity breakdown:")
print(df["race_ethnicity"].value_counts())
print(f"\nGPA summary:")
print(df[["grade8_overall_gpa","grade8_math_gpa","grade8_science_gpa"]].describe().round(2))

df.to_csv("synthetic_students.csv", index=False)
print(f"\nSaved: synthetic_students.csv")
