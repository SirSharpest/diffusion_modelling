import numpy as np
import json as js
import pandas as pd

with open('./data.json') as f:
    data = js.load(f)

rows = []
# Turn into a flattened table
for chem in data.keys():
    for p in data[chem].keys():
        for tp in data[chem][p]:
            for n, c in enumerate(data[chem][p][tp]):
                rows.append({'chemical': chem,
                             'pd_permiability': int(float(p.rsplit('_')[-1])*100),
                             'seconds': int(tp),
                             'cell_number': int(n),
                             'concentration': c})
df = pd.DataFrame(rows)
df.to_json('flat_Data.json')
