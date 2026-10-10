"""Writes a small MOCK dataset in the exact IMS file format, only to test ims_loader.py end to end.
Not real data. Usage: python make_mock_ims.py <out_dir>"""
import os, sys, datetime as dt
import numpy as np
out = sys.argv[1]
rng = np.random.default_rng(0)
SIGMA = {"1st_test": 0.1, "2nd_test": 0.35, "3rd_test": 0.05}   # different sensor scale per test, like real IMS
specs = [("1st_test", dt.datetime(2003, 10, 22, 12, 6, 24), 8, 4, 520, 0.65),
         ("2nd_test", dt.datetime(2004, 2, 12, 10, 32, 39), 4, 0, 510, 0.5),
         ("3rd_test", dt.datetime(2004, 3, 4, 9, 27, 46), 4, 2, 560, 0.75)]
for folder, t0, n_ch, bad, n_files, onset_frac in specs:
    os.makedirs(os.path.join(out, folder), exist_ok=True)
    onset = int(n_files * onset_frac)
    for i in range(n_files):
        t = t0 + dt.timedelta(minutes=10 * i)
        x = rng.normal(0, SIGMA[folder], (4096, n_ch))
        if i >= onset:
            p = (i - onset) / (n_files - onset)
            x[:, bad] *= 1 + 3 * p
            idx = rng.integers(0, 4096, int(1 + 8 * p)); x[idx, bad] += rng.normal(0, 5 * SIGMA[folder] * (1 + 3 * p), len(idx))
        np.savetxt(os.path.join(out, folder, t.strftime("%Y.%m.%d.%H.%M.%S")), x, fmt="%.4f", delimiter="\t")
print("mock written to", out)
