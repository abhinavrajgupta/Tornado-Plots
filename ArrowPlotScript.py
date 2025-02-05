import os
import cv2
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tensorflow
import seaborn as sns
import imutils
from math import dist
import pandas as pd

def get_module_paths(module_name):
    base_dir = os.path.join('/home/hgc-qc-web/Web/', 'Modules')
    images_dir = os.path.join(base_dir, module_name, 'ResultsArrowPlots')
    labels_dir = os.path.join(base_dir, module_name, 'ResultsArrowPlots','labels')
    return images_dir, labels_dir

def process_labels(label_path, img_path):
    dict1 = {}
    for name in os.listdir(img_path):
        if ".jpg" in name:
            try:
                image_path = os.path.join(img_path, name)
                mask = cv2.imread(image_path)
                (h, w) = mask.shape[:2]
                txtname = name.split('.jpg')[0] + ".txt"
                label_file_path = os.path.join(label_path, txtname)
                with open(label_file_path, 'r') as filename:
                    lines = filename.readlines()
                    (x, y) = ((lines[0].split(" "))[1], (lines[0].split(" "))[2])
                    point = (int(float(x) * w), int(float(y) * h))
                    dict1[name] = [point[0], point[1]]
            except :
                pass
    return dict1

def detect_hough_circles(img_path):
    dict2 = {}  
    for name in os.listdir(img_path):
        if ".jpg" in name:
            image_path = os.path.join(img_path, name)
            mask = cv2.imread(image_path)
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray, 5)
            circles = cv2.HoughCircles(
                blurred, 
                cv2.HOUGH_GRADIENT, dp=1, minDist=150, param1=100, param2=50, minRadius=90, maxRadius=110
            )
            if circles is not None:
                detected_circles = np.uint16(np.around(circles))  
                for (x, y, r) in detected_circles[0, :]:
                    dict2[name] = [[x, y], r]  
    return dict2

def synchronize_dicts(dict1, dict2):
    # Identify keys that are in dict1 but not in dict2
    keys_to_drop = [key for key in dict1 if key not in dict2]
    for key in keys_to_drop:
        del dict1[key]

    # Identify keys that are in dict2 but not in dict1
    keys_to_drop = [key for key in dict2 if key not in dict1]
    for key in keys_to_drop:
        del dict2[key]

    return dict1, dict2

def process_and_generate_df(img_path, label_path):
    dict1 = process_labels(label_path, img_path)  
    dict2 = detect_hough_circles(img_path)
    dict1, dict2 = synchronize_dicts(dict1, dict2)

    # Extract data from dict1 and dict2 and compute distances, angles
    X2 = [values[0] for values in dict1.values()]  # Extract X values from dict1
    Y2 = [values[1] for values in dict1.values()]  # Extract Y values from dict1

    X1 = [values[0][0] for values in dict2.values()]  # Extract X values from dict2
    Y1 = [values[0][1] for values in dict2.values()]  # Extract Y values from dict2

    # Calculate delta_X, delta_Y, and r (distance)
    changeinx = []
    changeiny = []
    r = []

    for x1, y1, x2, y2 in zip(X1, Y1, X2, Y2):
        delta_x = x2 - x1
        delta_y = y2 - y1
        distance = math.sqrt(delta_x**2 + delta_y**2)
        changeinx.append(delta_x)
        changeiny.append(delta_y)
        r.append(distance)

    # Calculate angles (in degrees)
    angles = []
    for delta_x, delta_y in zip(changeinx, changeiny):
        angle_in_radians = math.atan2(delta_y, delta_x)
        angle_in_degrees = math.degrees(angle_in_radians)
        angles.append(angle_in_degrees)

    # Prepare the image names list (from dict2, assuming filenames are the same)
    Images = list(dict2.keys())

    # Step 5: Create and return DataFrame
    df1 = pd.DataFrame({
        "Images": Images,
        "X1": X1,
        "Y1": Y1,
        "X2": X2,
        "Y2": Y2,
        "Delta_X": changeinx,
        "Delta_Y": changeiny,
        "r": r,
        "Angles": angles
    })
    df1 = sort_dataframe_by_image_number(df1)
    return df1

def sort_dataframe_by_image_number(df1):
    df2 = df1.copy()
    df2['ImageNumber'] = df2['Images'].str.extract('(\d+)').astype(int)
    df2 = df2.sort_values(by='ImageNumber')
    df3=df2.copy()
    df3=df3.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    #df2 = df2.drop(columns=['ImageNumber'])
    return df2,df3

def process_files_and_append_to_df(images_path,df3):
    working_dir = os.path.abspath(os.path.join(images_path, '..'))
    print(working_dir)
    os.chdir(working_dir)
    dict3 = {}
    for name in os.listdir():
        if ".txt" in name:
            with open(name, 'r') as filename:
                lines = filename.readlines()
                (x,y)=((lines[0].split(",")[0]),(lines[0].split(",")[1]))
                point=(float(x),float(y))
                dict3[name]=[point[0],point[1]] 
    x_values = []
    y_values = []
    for values in dict3.values():
        x_values.append(values[0])
        y_values.append(values[1])

    df3 = pd.DataFrame({
        "Images": list(dict3.keys()),
        "X": x_values,
        "Y": y_values,
    })
    df3['ImageNumber'] = df3['Images'].str.extract('(\d+)').astype(int)
    df3_sorted = df3.sort_values(by='ImageNumber')
    df3_sorted=df3_sorted[df3_sorted['ImageNumber'].isin(df3['ImageNumber'])]
    df3_sorted = df3_sorted.reset_index(drop=True)
    df3_sorted = df3_sorted.drop(columns=['ImageNumber'])
    return df3_sorted

def finaldf(df2, df3_sorted):
    finaldf = df2.merge(df3_sorted[['X','Y']], left_on='ImageNumber', right_index=True, how='left')
    return finaldf


def create_arrow_plot(finaldf, df3, images_path):
    x = np.array(finaldf['X'])
    y = np.array(finaldf['Y'])
    angles = np.array(finaldf['Angles'])
    r = np.array(finaldf['r'])
    
    fig, ax = plt.subplots()

    U = [r_value * math.cos(math.radians(-angle)) for angle, r_value in zip(angles, r)]
    V = [r_value * math.sin(math.radians(-angle)) for angle, r_value in zip(angles, r)]

    for i in range(len(U)):
        ax.quiver(x[i], y[i], U[i], V[i], scale=110, color='b', width=0.008)

    for i, label in enumerate(df3['ImageNumber']):
        plt.annotate(label, (x[i], y[i]), color='red', weight='bold', ha='center')

    ax.set_xlim(-60, 150)
    ax.set_ylim(-20, 240)
    ax.set_xticks([])
    ax.set_yticks([])

    reference_arrow = finaldf[finaldf['r'] == finaldf['r'].max()].index[0]

    arrow_length = math.sqrt(U[reference_arrow]**2 + V[reference_arrow]**2)

    ax.quiver(x[reference_arrow], y[reference_arrow], U[reference_arrow], V[reference_arrow], scale=110, color='g', width=0.008)

    bottom_left_x, bottom_left_y = -40, 180  # Adjust these values as needed
    ax.quiver(bottom_left_x, bottom_left_y, arrow_length, 0, scale=110, color='g', width=0.008)

    ax.annotate(f'Length: {(arrow_length) * 2 / 103:.2f}mm', (bottom_left_x + arrow_length, bottom_left_y + 5), color='red', fontsize=8, weight='bold', ha='center')

    ax.set_title('Combined Arrowplots')
    ax.grid(True)
    plt.tight_layout()
    save_path = os.path.join(images_path, "Arrowplot.jpg")
    plt.savefig(save_path, dpi=100)
    plt.show()

def main():
    module_name = input("Enter the module name (e.g., 'M35'): ")
    images_path, labels_path = get_module_paths(module_name)
    df1, df3 = process_and_generate_df(images_path, labels_path)
    df3_sorted = process_files_and_append_to_df(images_path,df3)
    final_df = finaldf(df1, df3_sorted)
    create_arrow_plot(final_df, df3, images_path)
    print("Arrowplots Saved Successfully!")

if __name__ == "__main__":
    main()

