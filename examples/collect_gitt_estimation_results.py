import json
# import numpy as np
# from contextlib import redirect_stdout

"""
tab = "    "
for choice in ("results",):#, "verification"):
    with open("../GITT estimation " + choice + "/estimation_" + choice + ".py",
              'w') as d:
        with redirect_stdout(d):
            print("import numpy as np")
            print()
            print("#list of estimations, correlation matrices and errorbars")
            print()
            print("estimation_" + choice + " = [")
            for i in range(84 + 1):
                with open("../GITT estimation " + choice + "/pulses/pulse_"
                          + str(i) + ".json", 'r') as f:
                    estimate, correlation, errorbars = json.load(f)
                    print(1 * tab + "[")
                    print(2 * tab + "{")
                    print(3 * tab + repr(estimate)[1:-1].replace(
                        ", ", ",\n" + 3 * tab
                    ).replace(": ", ":\n" + 4 * tab))
                    print(2 * tab + "},")
                    print(2 * tab + "np."
                          + repr(np.array(correlation)).replace(
                              "],\n", "],\n" + 2 * tab + "   "
                          ) + ",")
                    print(2 * tab + "{")
                    print(3 * tab + repr(errorbars)[1:-1].replace(
                        "], ", "],\n" + 3 * tab
                    ).replace(": ", ":\n" + 4 * tab))
                    print(2 * tab + "}")
                    print(1 * tab + "],")
            print("]")
            print()
"""

for choice in ("results", "verification"):
    collected_data = []
    for i in range(84 + 1):
        with open("../GITT estimation " + choice + "/pulses/pulse_"
                  + str(i) + ".json", 'r') as f:
            collected_data.append(json.load(f))
    with open("../GITT estimation " + choice + "/estimation_" + choice
              + ".json", 'w') as d:
        json.dump(collected_data, d)
