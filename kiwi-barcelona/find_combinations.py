import csv
import sys

f = sys.stdin.read().splitlines()
lines = csv.reader(f, delimiter=';')
lines = list(lines)
column_names = lines[0]
rows = lines[1:]
flights = pd.DataFrame(columns=column_names)
for i in range(len(rows)):
    flights.loc[i] = rows[i]

print(flights)
