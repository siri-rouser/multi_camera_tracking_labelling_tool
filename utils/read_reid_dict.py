'''
This script reads a dictionary from a file and prints its contents.
'''

ID_COUNT = 0 

import pandas as pd
import os
import json

def format_detection_line(cam_num, info, id_index):
    """
    Formats a detection into a line with the following fields:
      camera_id, obj_id, frame_id, xmin, ymin, width, height, xworld, yworld
    Expects 'info' to be a tuple: (frame, [x1, y1, x2, y2], cls).
    Here, xworld and yworld are fixed to -1.
    """
    frame_num = int(info[0])
    bbox = info[1]
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    xworld = -1
    yworld = -1
    line = f"{cam_num} {id_index} {frame_num} {x1:.2f} {y1:.2f} {width:.2f} {height:.2f} {xworld} {yworld}"
    return line

def read_tracks(cam_id):
    data = {}
    filepath = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/imagesc00{cam_id}/imagesc00{cam_id}_mot_interpolated_final.txt'
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        data = {}
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
                if track_id not in data:
                    data[track_id] = []
                data[track_id].append((frame, [x1, y1, x2, y2], cls))
    return data

def read_data(df, cam_id1, cam_id2, name, reid_dict): 
    # e.g 1->2, cam_id1 is 1, cam_id2 is 2
    """
    Reads data from a DataFrame and updates the reid_dict with track IDs and their corresponding new IDs.
    """
    global ID_COUNT

    # initialize the reid_dict
    if cam_id1 not in reid_dict[name]:
        reid_dict[name][cam_id1] = {}
    if cam_id2 not in reid_dict[name]:
        reid_dict[name][cam_id2] = {}

    for index, track_id in df['Inlet'].items():
        # Skip if track_id is None or Exit is None
        if pd.isna(track_id) or pd.isna(df['Exit'][index]):
            print(f"Skipping track id:{track_id} in {cam_id1} and track id:{df['Exit'][index]} in {cam_id2} due to None values in track_id or Exit.")
            continue

        if track_id not in reid_dict[name][cam_id1]:
            reid_dict[name][cam_id1][track_id] = {}

        reid_dict[name][cam_id1][track_id]['ori_id'] = track_id
        reid_dict[name][cam_id1][track_id]['new_id'] = ID_COUNT
        
        if df['Exit'][index] not in reid_dict[name][cam_id2]:
            reid_dict[name][cam_id2][df['Exit'][index]] = {}

        reid_dict[name][cam_id2][df['Exit'][index]]['ori_id'] = df['Exit'][index]
        reid_dict[name][cam_id2][df['Exit'][index]]['new_id'] = ID_COUNT
        ID_COUNT += 1

        if ID_COUNT % 50 == 0:
            print(f"Processed {ID_COUNT} IDs so far.")

def reid_dict_cat_backup(dict1, dict2):
    global ID_COUNT
    merged = dict1.copy()
    

    for cam_id, tracks2 in dict2.items():
        if cam_id not in merged:
            print(f"Camera ID {cam_id} not found in merged dict, adding it.")
            merged[cam_id] = tracks2.copy()
        else:
            print(f"Camera ID {cam_id} found in merged dict, merging tracks.")
            for track_id, data2 in tracks2.items():
                if track_id in merged[cam_id]:
                    # If track_id overlaps, assign same new_id
                    new_id = ID_COUNT
                    merged[cam_id][track_id]['new_id'] = new_id
                    dict2[cam_id][track_id]['new_id'] = new_id
                    ID_COUNT += 1
                else:
                    # No overlap, just copy it
                    merged[cam_id][track_id] = data2.copy()
    return merged

def reid_dict_cat(dict1,dict2):
    global ID_COUNT

    overlap = {}
    for cam_id in dict2:
        if cam_id in dict1:
            print(f"Camera ID {cam_id} found in both dictionaries, checking for overlaps.")
            overlap[cam_id] = {}

    for cam_id in dict2:
        if cam_id in list(dict1.keys()):
            for track_id in dict2[cam_id]:
                if track_id in list(dict1[cam_id].keys()):
                    overlap[cam_id][track_id] = track_id

    for cam_id1 in overlap:
        for track_id in overlap[cam_id1]:
            if track_id in list(dict1[cam_id1].keys()) and track_id in list(dict2[cam_id1].keys()):
                print(f"Track ID {track_id} overlaps in Camera ID {cam_id1}, assigning same new_id.")
                # If track_id overlaps, assign same new_id
                dict1_new_id = dict1[cam_id1][track_id]['new_id']
                dict2_new_id = dict2[cam_id1][track_id]['new_id'] 
                for cam_id2 in dict1:
                    for track_id in dict1[cam_id2]:
                        if dict1[cam_id2][track_id]['new_id'] == dict1_new_id:
                            dict1[cam_id2][track_id]['new_id'] = ID_COUNT
                            break

                for cam_id3 in dict2:
                    for track_id in dict2[cam_id3]:
                        if dict2[cam_id3][track_id]['new_id'] == dict2_new_id:
                            dict2[cam_id3][track_id]['new_id'] = ID_COUNT
                            break
                ID_COUNT += 1

    merged = dict1.copy()
    for cam_id, tracks2 in dict2.items():
        if cam_id not in merged:
            print(f"Camera ID {cam_id} not found in merged dict, adding it.")
            merged[cam_id] = tracks2.copy()
        else:
            print(f"Camera ID {cam_id} found in merged dict, merging tracks.")
            for track_id, data2 in tracks2.items():
                if track_id not in merged[cam_id]:
                    merged[cam_id][track_id] = data2.copy()
                else:
                    print(f"Track ID {track_id} already exists in Camera ID {cam_id}, skipping.")
                    pass

    return merged

def vehicle_number_count(reid_dict):
    """
    Counts the number of unique vehicles in the reid_dict and prints the count.
    """
    Atracks = []
    total_vehicles = 0
    for cam_id, tracks in reid_dict.items():
        for track_id, data in tracks.items():
            if data['new_id'] not in tracks:
                Atracks.append(data['new_id'])
        total_vehicles += len(tracks)
        print(f"Camera ID {cam_id} has {len(tracks)} unique vehicles.")

    print(f"Total unique vehicles across all cameras: {len(Atracks)}")

if __name__ == "__main__":
    '''
    Main function to read cross camera information from an Excel file and format it into a txt ground truth file.
    '''
    file_path = '../temp_res/Vehicle Tracking Final copy.xlsx'
    reid_dict = {}
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    # View all sheet names
    print(xls.sheet_names)

    for sheetname in sheet_names:
        print(f"Reading sheet: {sheetname}")
    
        # Read the sheet into a DataFrame
        df = pd.read_excel(xls, sheet_name=f'{sheetname}')

        # if the sheet name is 2->3, the cam_id1 is 2
        cam_id1 = int(sheetname[0]) # inlet
        cam_id2 = int(sheetname[-1]) # exit
        print(f"Processing pair: {cam_id1} -> {cam_id2}")

        # Define the name of reid_dict
        if cam_id1 < cam_id2:
            name = f"{cam_id1}_{cam_id2}"
        else:
            name = f"{cam_id2}_{cam_id1}"

        if name not in reid_dict:
            reid_dict[name] = {}
        read_data(df, cam_id1, cam_id2, name, reid_dict)

    # DATA read complete---------------------------------------
    print('Data read complete.')
    # Print all keys in reid_dict
    print("Keys in reid_dict:")
    for key in reid_dict.keys():
        print(key)

    final_redict = reid_dict['1_2']

    # merge reid_dict[key] into final_redict
    for key in reid_dict.keys():
        if key != '1_2':
            print(f"Merging {key} into final_redict")
            final_redict = reid_dict_cat(final_redict, reid_dict[key])

    with open('final_redict.json', 'w') as f:
        json.dump(final_redict, f, indent=2)
    print("reid_dict merge complete. . . ")
    print('Writing to Multi_CAM_Ground_Turth.txt...')

    vehicle_number_count(final_redict)

    with open('Multi_CAM_Ground_Turth.txt', 'w') as f:
        for cam_id, tracks in final_redict.items():
            print(f"Camera ID: {cam_id}")
            trackletdata = read_tracks(cam_id)
            for track_id, data in tracks.items():
                ori_id = data['ori_id']
                new_id = data['new_id']
                if ori_id in trackletdata:
                    info = trackletdata[ori_id]
                    for item in info:
                        line = format_detection_line(cam_id, item, new_id)
                        f.write(line + '\n')