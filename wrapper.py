# wrapper.py

import subprocess
import time
import sys
import os
from io import StringIO

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Get the API key from user input
GROK_API_KEY = input("Enter your GROK_API_KEY: ").strip()

# Set the environment variable
os.environ['GROK_API_KEY'] = GROK_API_KEY

# Define script names
first_script = '_1_funnel.py'
second_script = '_2_grade_and_summarize.py'

# Run the first script and capture output while printing
print(f"Running {first_script}...")
process1 = subprocess.Popen([sys.executable, first_script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

buf = StringIO()
tee = Tee(sys.stdout, buf)

for line in process1.stdout:
    tee.write(line)

process1.wait()

output = buf.getvalue()

# Check if "DONEZO" was printed
if "DONEZO" in output:
    print("First script completed successfully.")
    time.sleep(1)
    
    # Run the second script
    print(f"Running {second_script}...")
    process2 = subprocess.Popen([sys.executable, second_script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process2.stdout:
        sys.stdout.write(line)
    
    process2.wait()
    
    print("Second script completed.")
else:
    print("First script did not print 'DONEZO'. Second script not executed.")