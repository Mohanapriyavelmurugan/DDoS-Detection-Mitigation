import os

files = ['measure_latency.py', 'measure_throughput.py', 'measure_offloading.py', 'measure_scalability.py']

for fn in files:
    if os.path.exists(fn):
        with open(fn, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the open call for writing the output file to include utf-8
        content = content.replace('open(args.output, "w")', 'open(args.output, "w", encoding="utf-8")')
        
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed encoding in {fn}")
