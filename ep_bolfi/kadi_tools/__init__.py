"""
Each file in here is a CLI-executable tool, in the format:
 - python -m ep_bolfi.kadi_tools.<filename_without_.py>
Due to the way the CLI-executableness is implemented, the methods are
not directly callable the way they appear in the source code, so please
refer to their --help outputs, which are stored alongside documentation.

If you want to use the methods from inside a Python script, you need to
imitate the CLI execution, for example:

from subprocess import run
from ep_bolfi.utility.dataset_formatting import Cycling_Information
read_call = run([
    "python",
    "-m",
    "ep_bolfi.kadi_tools.read_measurement_from_parquet",
    "-r",
    "<record_id>",
    "-n",
    "<filename>",
], capture_output=True)
data = Cycling_Information.from_json(read_call.stdout)
"""
