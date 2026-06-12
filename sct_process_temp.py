import argparse
def main(input_file, output_file):
    with open(input_file, 'r') as f:
        with open(output_file, 'w') as fw:
            for line in f:
                parts = line.strip().split()

                frame_id = int(parts[0])
                track_id = int(parts[1])
                x1 = (float(parts[2]) / 2160) * 3840
                y1 = (float(parts[3]) / 3840) * 2160
                x2 = (float(parts[4]) / 2160) * 3840
                y2 = (float(parts[5]) / 3840) * 2160

                line = f'{frame_id} {track_id} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n'
                fw.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tracking sequences.")
    parser.add_argument('--input_file', type=str, help="Input tracking file.")
    parser.add_argument('--output_file', type=str, help="Output file for interpolated results.")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    main(input_file, output_file)
