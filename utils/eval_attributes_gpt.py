import pandas as pd
import os
import json
from collections import defaultdict

# ---------------- Config ----------------
EXCEL_PATH = '../temp_res/Vehicle Tracking Final copy.xlsx'
TRAJ_DIR_TMPL = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/imagesc00{cam}/imagesc00{cam}_mot_interpolated_final.txt'
ADJACENT_PAIRS = [(1, 2), (2, 3), (3, 4)]  # We'll also consider reverse automatically
TRANSIT_MAX_FRAMES = 6000
OUTPUT_TXT = 'multi_cam_match_rlbased.txt'

# ------------- Helpers -------------
def _norm_id(x):
    if pd.isna(x):
        return None
    try:
        return int(x)
    except Exception:
        return str(x).strip()

def _norm_attr(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().lower()
    return s if s != '' else None

def read_traj_txt(cam_id):
    traj_data = {}
    filepath = TRAJ_DIR_TMPL.format(cam=cam_id)
    if not os.path.exists(filepath):
        print(f"[WARN] File {filepath} does not exist.")
        return {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x1 = float(parts[2]); y1 = float(parts[3])
            x2 = float(parts[4]); y2 = float(parts[5])
            cls = int(float(parts[6]))
            traj_data.setdefault(track_id, []).append((frame, [x1, y1, x2, y2], cls))
    return traj_data

def format_line(cam_num, frame, bbox, global_id):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return f"{cam_num} {global_id} {int(frame)} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f}"

def read_xlsx_attributes(xls):
    """
    Build a dict: data[cam_id][track_id] = {'make':..., 'model':..., 'colour':...}
    from all sheets like '1->2', '2->3', etc.
    """
    data = {}
    for sheetname in xls.sheet_names:
        print(f"[Excel] Reading sheet: {sheetname}")
        df = pd.read_excel(xls, sheet_name=sheetname)
        df = df.copy()
        df.columns = df.columns.astype(str).str.strip()

        cam_id1 = int(sheetname[0])   # inlet
        cam_id2 = int(sheetname[-1])  # exit

        for idx, inlet_id in df['Inlet'].items():
            exit_id = df.loc[idx, 'Exit']
            colour  = _norm_attr(df.loc[idx, 'Colour'])
            make    = _norm_attr(df.loc[idx, 'Make'])
            model   = _norm_attr(df.loc[idx, 'Model'])

            id1 = _norm_id(inlet_id)
            id2 = _norm_id(exit_id)

            if id1 is not None:
                data.setdefault(cam_id1, {}).setdefault(id1, {})
                data[cam_id1][id1].update({'colour': colour, 'make': make, 'model': model})

            if id2 is not None:
                data.setdefault(cam_id2, {}).setdefault(id2, {})
                data[cam_id2][id2].update({'colour': colour, 'make': make, 'model': model})
    return data

# -------- Union-Find for global ID linking --------
class DSU:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def _key(self, cam_id, track_id):
        return (int(cam_id), int(track_id))

    def add(self, cam_id, track_id):
        k = self._key(cam_id, track_id)
        if k not in self.parent:
            self.parent[k] = k
            self.rank[k] = 0

    def find(self, cam_id, track_id):
        k = self._key(cam_id, track_id)
        if k not in self.parent:
            self.add(cam_id, track_id)
        # path compression
        if self.parent[k] != k:
            self.parent[k] = self.find(*self.parent[k])
        return self.parent[k]

    def union(self, a_cam, a_id, b_cam, b_id):
        ra = self.find(a_cam, a_id)
        rb = self.find(b_cam, b_id)
        if ra == rb:
            return
        # union by rank
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

# ------------- Main logic -------------
if __name__ == "__main__":
    # 1) Load attributes from Excel
    xls = pd.ExcelFile(EXCEL_PATH)
    data = read_xlsx_attributes(xls)

    # 2) Load trajectories, compute min/max frames
    for cam_id in list(data.keys()):
        traj = read_traj_txt(cam_id)
        for tid, attr in data[cam_id].items():
            track_traj = traj.get(int(tid), [])
            attr['trajectory'] = track_traj
            if track_traj:
                frames = [t[0] for t in track_traj]
                attr['min_frame'] = min(frames)
                attr['max_frame'] = max(frames)
            else:
                attr['min_frame'] = None
                attr['max_frame'] = None

    # 3) Build DSU and link across adjacent cameras if:
    #    - attribute match (make/model/colour all equal where known)
    #    - time transit within TRANSIT_MAX_FRAMES for forward or reverse
    dsu = DSU()
    # Ensure all seen tracks are in DSU
    for cam_id, tracks in data.items():
        for tid in tracks.keys():
            dsu.add(cam_id, tid)

    def attrs_match(a, b):
        # Equal if both not None then they must be equal; None acts like wildcard (your original rule)
        for k in ('make', 'model', 'colour'):
            va = a.get(k); vb = b.get(k)
            if va is not None and vb is not None and va != vb:
                return False
        return True

    # function to test transit condition; FIXED the sign from your original snippet
    def transit_ok(a, b):
        a_min, a_max = a.get('min_frame'), a.get('max_frame')
        b_min, b_max = b.get('min_frame'), b.get('max_frame')
        if a_min is None or a_max is None or b_min is None or b_max is None:
            return False
        # forward A->B: B appears shortly AFTER A
        if 0 < (b_min - a_max) < TRANSIT_MAX_FRAMES:
            return True
        # reverse B->A: A appears shortly AFTER B
        if 0 < (a_min - b_max) < TRANSIT_MAX_FRAMES:
            return True
        return False

    # Check both directions for each adjacent pair
    for (c1, c2) in ADJACENT_PAIRS:
        for cam_a, cam_b in [(c1, c2), (c2, c1)]:
            if cam_a not in data or cam_b not in data:
                continue
            for id_a, attr_a in data[cam_a].items():
                for id_b, attr_b in data[cam_b].items():
                    if not attrs_match(attr_a, attr_b):
                        continue
                    if not transit_ok(attr_a, attr_b):
                        continue
                    dsu.union(cam_a, id_a, cam_b, id_b)

    # 4) Assign compact global IDs
    root_to_gid = {}
    next_gid = 0
    camid_tid_to_gid = {}
    for cam_id, tracks in data.items():
        for tid in tracks.keys():
            root = dsu.find(cam_id, tid)
            if root not in root_to_gid:
                root_to_gid[root] = next_gid
                next_gid += 1
            camid_tid_to_gid[(cam_id, tid)] = root_to_gid[root]

    print(f"[Info] Assigned {next_gid} global IDs.")

    # 5) Write results to TXT in the required format:
    #    {cam_num} {id_index} {frame_num} {x1:.2f} {y1:.2f} {width:.2f} {height:.2f}
    #    We follow the same bbox-to-line formatting convention you used in your reference script. :contentReference[oaicite:0]{index=0}
    with open(OUTPUT_TXT, 'w') as f:
        total_written = 0
        for cam_id, tracks in sorted(data.items()):
            for tid, attr in tracks.items():
                gid = camid_tid_to_gid[(cam_id, tid)]
                traj = attr.get('trajectory', [])
                for (frame, bbox, _cls) in traj:
                    f.write(format_line(cam_id, frame, bbox, gid) + '\n')
                    total_written += 1

    print(f"[Done] Wrote {total_written} lines to {OUTPUT_TXT}")
