import csv
from collections import defaultdict, deque

# Load graph
concepts = {}
dependencies = defaultdict(list)

with open('learning-graph.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cid = int(row['ConceptID'])
        concepts[cid] = row['ConceptLabel']
        if row['Dependencies']:
            deps = [int(d) for d in row['Dependencies'].split('|')]
            dependencies[cid] = deps

# Check for self-dependencies
self_deps = []
for cid, deps in dependencies.items():
    if cid in deps:
        self_deps.append((cid, concepts[cid]))

if self_deps:
    print('Self-dependencies found:')
    for cid, label in self_deps:
        print(f'  {cid}: {label}')
else:
    print('No self-dependencies found')

# Build reverse graph (who depends on me)
reverse_deps = defaultdict(list)
for cid, prereqs in dependencies.items():
    for prereq in prereqs:
        reverse_deps[prereq].append(cid)

# Kahn's algorithm
indeg = {cid: 0 for cid in concepts}
for cid, prereqs in dependencies.items():
    indeg[cid] = len(prereqs)

queue = deque([cid for cid in concepts if indeg[cid] == 0])
processed = []

while queue:
    node = queue.popleft()
    processed.append(node)
    for dependent in reverse_deps[node]:
        indeg[dependent] -= 1
        if indeg[dependent] == 0:
            queue.append(dependent)

print(f'Total concepts: {len(concepts)}')
print(f'Processed: {len(processed)}')
print(f'Is DAG: {len(processed) == len(concepts)}')

if len(processed) != len(concepts):
    unprocessed = [cid for cid in concepts if cid not in processed]
    print(f'Unprocessed concepts ({len(unprocessed)}): {unprocessed[:10]}')
    print('This indicates a cycle in the graph.')
