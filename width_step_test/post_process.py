import re
import csv
import os

def extract_data(file_content):
    pattern = r"Running Layer \d+\s*\nCompute cycles: (\d+)\s*\nStall cycles: (\d+)\s*\n\s*IFM Stall cycles: ([\d.]+)\s*\n\s*Filter Stall cycles: ([\d.]+)\s*\nOverall utilization: ([\d.]+)%\s*\nMAC Core Mapping efficiency: ([\d.]+)%\s*\nAverage IFMAP SRAM BW: ([\d.]+) bytes/cycle\s*\nAverage Filter SRAM BW: ([\d.]+) bytes/cycle\s*\nAverage OFMAP SRAM BW: ([\d.]+) bytes/cycle"

    result = re.findall(pattern, file_content)
    return result

def extract_suffix_number(filename):
    pattern = r'\d+'
    match = re.search(pattern, filename)
    if match:
        return int(match.group())
    else:
        return None

filename_prefix = "width_step_"
csv_header = ["File", "Compute cycles", "Stall cycles", "IFM Stall cycles", "Filter Stall cycles", "Overall utilization", "MAC Core Mapping efficiency", "Average IFMAP SRAM BW", "Average Filter SRAM BW", "Average OFMAP SRAM BW"]

csv_data_layer0 = []
csv_data_layer1 = []

files = [file for file in os.listdir('.') if file.endswith('.log')]

for file in files:
    with open(file, 'r') as f:
        content = f.read()
        data = extract_data(content)
        suffix_number = extract_suffix_number(file)
        if data:
            for i, layer_data in enumerate(data):
                layer = f"Layer {i}"
                row_data = [file] + list(layer_data)
                if "Layer 0" in layer:
                    csv_data_layer0.append(row_data)
                elif "Layer 1" in layer:
                    csv_data_layer1.append(row_data)

# 对数据按后缀数字进行排序
csv_data_layer0 = sorted(csv_data_layer0, key=lambda x: extract_suffix_number(x[0]))
csv_data_layer1 = sorted(csv_data_layer1, key=lambda x: extract_suffix_number(x[0]))

with open('layer0_data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)
    csv_writer.writerows(csv_data_layer0)

with open('layer1_data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)
    csv_writer.writerows(csv_data_layer1)
