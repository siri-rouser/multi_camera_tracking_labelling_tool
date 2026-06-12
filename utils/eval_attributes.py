import pandas as pd
import os
import json

ID_COUNT = 0

def _norm_id(x):
    if pd.isna(x):
        return None
    try:
        return int(x)  # turn 14.0 -> 14
    except Exception:
        return str(x).strip()

def read_data(df, cam_id1, cam_id2, data):
    # make sure column names are clean
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    for idx, inlet_id in df['Inlet'].items():
        exit_id = df.loc[idx, 'Exit']          # same row, label-safe
        colour  = df.loc[idx, 'Colour']
        make    = df.loc[idx, 'Make']
        model   = df.loc[idx, 'Model']

        print(f"Processing row {idx}: Inlet ID {inlet_id}, Exit ID {exit_id}, "
              f"Colour {colour}, Make {make}, Model {model}")

        id1 = _norm_id(inlet_id)
        id2 = _norm_id(exit_id)

        if id1 is not None:
            data.setdefault(cam_id1, {})
            data[cam_id1][id1] = {'colour': colour, 'make': make, 'model': model}

        if id2 is not None:
            data.setdefault(cam_id2, {})
            data[cam_id2][id2] = {'colour': colour, 'make': make, 'model': model}

    return data

def read_traj_txt(cam_id):
    traj_data = {}
    filepath = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/imagesc00{cam_id}/imagesc00{cam_id}_mot_interpolated_final.txt'
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        traj_data = {}
    else:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x1 = float(parts[2])
                y1 = float(parts[3])
                x2 = float(parts[4])
                y2 = float(parts[5])
                cls = int(float(parts[6]))
                if track_id not in traj_data:
                    traj_data[track_id] = []
                traj_data[track_id].append((frame, [x1, y1, x2, y2], cls))
    return traj_data

if __name__ == "__main__":
    '''
    Main function to read cross camera information from an Excel file and format it into a txt ground truth file.
    '''
    file_path = '../temp_res/Vehicle Tracking Final copy.xlsx'
    data = {}
    reid_dict = {}
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    # View all sheet names
    print(xls.sheet_names)

    # Read each sheet and insert the insert the colour/make/model information into the data dict
    for sheetname in sheet_names:
        print(f"Reading sheet: {sheetname}")
    
        # Read the sheet into a DataFrame
        df = pd.read_excel(xls, sheet_name=f'{sheetname}')

        # if the sheet name is 2->3, the cam_id1 is 2
        cam_id1 = int(sheetname[0]) # inlet
        cam_id2 = int(sheetname[-1]) # exit
        print(f"Processing pair: {cam_id1} -> {cam_id2}")

        data = read_data(df, cam_id1, cam_id2, data)

    # Read the trajectory information from the txt files and insert them into the data dict
    for cam_id, tracks in data.items():
        trajectory_data = read_traj_txt(cam_id)
        for track_id, attributes in tracks.items():
            if track_id in trajectory_data: # NOTE: potential bug if track_id not in the same data type
                attributes['trajectory'] = trajectory_data[track_id]
                attributes['max_frame'] = max([item[0] for item in trajectory_data[track_id]])
                attributes['min_frame'] = min([item[0] for item in trajectory_data[track_id]])
            else:
                attributes['trajectory'] = []
                attributes['max_frame'] = None
                attributes['min_frame'] = None

    # Do Multi-Camera Vehicle Matching
    cam_list = [1,2,3]
    for cam_id in cam_list:
        if cam_id in data:
            for track_id1, attributes1 in data[cam_id].items():
                max_frame1 = attributes1.get('max_frame', None)
                min_frame1 = attributes1.get('min_frame', None)
                make1 = attributes1.get('make', None)
                model1 = attributes1.get('model', None)
                colour1 = attributes1.get('colour', None)
                for track_id2, attributes2 in data[cam_id+1].items():
                    max_frame2 = attributes2.get('max_frame', None)
                    min_frame2 = attributes2.get('min_frame', None)
                    make2 = attributes2.get('make', None)
                    model2 = attributes2.get('model', None)
                    colour2 = attributes2.get('colour', None)

                    if not ((2000 > max_frame1 - min_frame2 > 0) or (2000 > max_frame2 - min_frame1 > 0)):
                        continue

                    if not ((make1 == make2 or make1 is None or make2 is None) and (model1 == model2 or model1 is None or model2 is None) and (colour1 == colour2 or colour1 is None or colour2 is None)):
                        continue

