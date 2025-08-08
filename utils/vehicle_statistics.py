import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os

# Ensure Times New Roman font globally
plt.rcParams['font.family'] = 'serif'

def colour_rename(colour):
    """
    Renames the colour to a more readable format.
    """
    if colour == 'Dark Blue':
        return 'Blue'
    elif colour == 'White ':
        return 'White'
    elif colour == 'Dark Red':
        return 'Red'
    elif colour == 'Red/Yellow':
        return 'Red'
    elif colour == 'White Teal Black':
        return 'White'
    elif colour == 'Teal':
        return 'Blue'
    elif colour == 'Teal ':
        return 'Blue'
    elif colour == 'Yellow Red':
        return 'Orange'
    elif colour == 'Golden':
        return 'Orange'
    elif colour == 'Gold':
        return 'Orange'
    elif colour == 'Dark Green':
        return 'Green'
    elif colour == 'white':
        return 'White'
    elif colour == 'White Blue Black':
        return 'White'
    elif colour == 'Red ':
        return 'Red'
    elif colour == 'White Green':
        return 'White'
    elif colour == 'Yellow':
        return 'Orange'
    elif colour == 'Silver':
        return 'Grey'
    else:
        return colour
    
def cat_rename(category):
    if category == 'SUV and Offroaders':
        return 'SUV'
    elif category == 'People Carriers and MPV':
        return 'MPV'
    elif category == 'Heavy Duty Trucks':
        return 'HGV'
    elif category == 'Small Passenger Vehicles':
        return 'Car'
    elif category == 'Commercial Van':
        return 'Van'
    elif category == 'Pick Up':
        return 'Pickup'
    else:
        print(f"Unknown category: {category}")
        return category
    
    
def draw_pie_charts(allinfo_dict, color_scheme='Set3'):
    """
    Draw pie charts for Colour and Category stats with legend, custom palette, and black borders.
    """
    def get_color_palette(n, cmap_name):
        # Use Tableau colors from matplotlib
        tableau_colors = list(mcolors.TABLEAU_COLORS.values())
        if n <= len(tableau_colors):
            return tableau_colors[:n]
        else:
            # If more colors are needed, repeat the list or use a colormap fallback
            cmap = cm.get_cmap(cmap_name)
            return [cmap(i / n) for i in range(n)]

    def draw_pie(data_dict, filename, b2a, preset_ncol=3, color_coding=False, title=None):
        labels = list(data_dict.keys())
        counts = list(data_dict.values())
        total = sum(counts)

        if color_coding:
            def map_to_named_color(label):
                c_norm = label.lower().strip()
                tableau_map = {
                    'white': 'white',
                    'black': 'black',
                    'red': mcolors.TABLEAU_COLORS['tab:red'],
                    'blue': mcolors.TABLEAU_COLORS['tab:blue'],
                    'yellow': mcolors.TABLEAU_COLORS['tab:orange'],
                    'green': mcolors.TABLEAU_COLORS['tab:green'],
                    'grey': mcolors.TABLEAU_COLORS['tab:gray'],
                    'gray': mcolors.TABLEAU_COLORS['tab:gray'],
                    'silver': mcolors.TABLEAU_COLORS['tab:gray'],
                    'orange': mcolors.TABLEAU_COLORS['tab:orange'],
                    'brown': mcolors.TABLEAU_COLORS['tab:brown'],
                    'purple': mcolors.TABLEAU_COLORS['tab:purple'],
                    'multicolour': mcolors.TABLEAU_COLORS['tab:pink'],
                }
                return tableau_map.get(c_norm)
            colors = [map_to_named_color(c) for c in labels]
        else:
            colors = get_color_palette(len(labels), color_scheme)

        fig, ax = plt.subplots(figsize=(8, 8))  # Fix figure size
        wedges, texts, autotexts = ax.pie(
            counts,
            radius=1,                      # Ensure both pies have the same radius
            colors=colors,
            startangle=140,
            autopct='',
            wedgeprops={'edgecolor': 'black'},
            textprops={'fontsize': 8},
        )

        ax.set_title(title, fontsize=12)
        ax.axis('equal')

        # Align axes position to standardize pie location
        ax.set_position([0.1, 0.25, 0.8, 0.65]) # [left, bottom, width, height]

        # Add legend outside the pie
        percentages = [count / total * 100 for count in counts]
        ax.legend(
            wedges,
            [f"{label} ({percent:.1f}%)" for label, count, percent in zip(labels, counts, percentages)],
            loc="lower center",
            bbox_to_anchor=(0.5, b2a),
            fontsize=16,
            ncol=preset_ncol,
            frameon=False
        )

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    # Draw Colour Chart (with name-based color mapping)
    draw_pie(allinfo_dict['Colour'], 'colour_distribution_pie_chart.png', b2a=-0.35,preset_ncol=2,color_coding=True)

    # Draw Category Chart (with palette-based coloring)
    draw_pie(allinfo_dict['Category'],  'category_distribution_pie_chart.png', b2a=-0.25, preset_ncol=2,color_coding=False)


if __name__ == "__main__":
    '''
    Main function to read vehicle statistics from an Excel file and draw pie charts.
    '''
    
    file_path = '../temp_res/Vehicle_statistics_mk3.xlsx'
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        exit(1)
    else:
        xls = pd.ExcelFile(file_path)
        info_dict = {}
        allinfo_dict = {}
        allinfo_dict['Colour'] = {}
        allinfo_dict['Category'] = {}
        allinfo_dict['Make'] = {}
        allinfo_dict['Model'] = {}
        allinfo_dict['vehicle_num'] = {}

    print("Available sheet names:", xls.sheet_names)

    for sheet_name in xls.sheet_names:
        info_dict[sheet_name] = {}
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        allinfo_dict['vehicle_num'][sheet_name] = len(df)
        print(f"Data from sheet '{sheet_name}':")
        print(df.head())

        df = pd.read_excel(file_path, sheet_name=sheet_name)

        for idx,colour in enumerate(df['Colour']):
            colour = colour_rename(colour)
            if colour not in info_dict[sheet_name]:
                info_dict[sheet_name][colour] = 0

            if colour not in allinfo_dict['Colour']:
                allinfo_dict['Colour'][colour] = 0
            info_dict[sheet_name][colour] += 1
            allinfo_dict['Colour'][colour] += 1

            cat = cat_rename(df['Category'][idx])

            if cat not in info_dict[sheet_name]:
                info_dict[sheet_name][cat] = 0

            if cat not in allinfo_dict['Category']:
                allinfo_dict['Category'][cat] = 0
            allinfo_dict['Category'][cat] += 1
            info_dict[sheet_name][cat] += 1

            if df['Make'][idx] not in allinfo_dict['Make']:
                allinfo_dict['Make'][df['Make'][idx]] = 0
            allinfo_dict['Make'][df['Make'][idx]] += 1

            if df['Model'][idx] not in allinfo_dict['Model']:
                allinfo_dict['Model'][df['Model'][idx]] = 0
            allinfo_dict['Model'][df['Model'][idx]] += 1



    print("Final statistics:")
    print(info_dict)


    print("All statistics:")
    print(allinfo_dict)

    # Print top 5 values of allinfo_dict['Make']
    top_5_makes = sorted(allinfo_dict['Make'].items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 Makes:")
    for make, count in top_5_makes:
        print(f"{make}: {count}")

    top_5_models = sorted(allinfo_dict['Model'].items(), key=lambda x: x[1], reverse=True)[:6]
    print("Top 5 Makes:")
    for model, count in top_5_models:
        print(f"{model}: {count}")

    draw_pie_charts(allinfo_dict)

    print(allinfo_dict['vehicle_num'])
    
