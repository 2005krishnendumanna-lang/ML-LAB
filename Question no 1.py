import csv
import random

# Define the file name
file_name = "student_marks.csv"

# Open the file in write mode with proper newline handling
with open(file_name, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(["Roll No", "Physics", "Chemistry", "Math"])

    # Loop to generate data for Roll No 1 to 100
    for roll_no in range(1, 101):
        # Generate random marks between 40 and 100
        physics = random.randint(40, 100)
        chemistry = random.randint(40, 100)
        math = random.randint(40, 100)

        # Write the row to the CSV file
        writer.writerow([roll_no, physics, chemistry, math])

print(f"Success! '{file_name}' has been created with 100 rows.")
