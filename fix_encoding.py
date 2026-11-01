import os

files = ['measure_latency.py', 'measure_throughput.py',
         'measure_offloading.py', 'measure_scalability.py']

UNICODE_MAP = [
    ('\u2500', '-'), ('\u2550', '='), ('\u2560', '+'), ('\u2563', '+'),
    ('\u256c', '+'), ('\u2566', '+'), ('\u2569', '+'), ('\u2554', '+'),
    ('\u2557', '+'), ('\u255a', '+'), ('\u255d', '+'), ('\u2551', '|'),
    ('\u255e', '+'), ('\u2561', '+'),
]

RECONFIGURE = 'import sys\nif hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")\n'

for fn in files:
    path = os.path.join('.', fn)
    text = open(path, encoding='utf-8').read()
    for uc, asc in UNICODE_MAP:
        text = text.replace(uc, asc)
    if 'sys.stdout.reconfigure' not in text:
        text = text.replace('warnings.filterwarnings("ignore")\n',
                            'warnings.filterwarnings("ignore")\n' + RECONFIGURE)
    open(path, 'w', encoding='utf-8').write(text)
    print(f'Fixed: {fn}')

print('All done.')
