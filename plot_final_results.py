import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

file = open("cross_validated_recognition_results", 'rb')
result = pickle.load(file)
print(result.get_box_plot_data())
file.close()

data = []
for i in range(0, 10):
    data.append(np.random.uniform(0.97, 0.999, 10))


def boxplot(data, title="Comparison of test accuracy across exercises"):
    numOfClasses = 10
    classes_names = ["Push ups", "Pull ups", "Burpees", "Deadlifts", "Box Jumps", "Air squats", "Situps", "Wall balss",
                     "KB Press", "KB Thrusters"]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title(title)
    ax1.set_xlabel('Exercise')
    ax1.set_ylabel('Test Accuracy')

    # Now fill the boxes with desired colors
    boxColors = ['darkkhaki', 'royalblue']
    medians = list(range(numOfClasses))
    for i in range(numOfClasses):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue
        k = i % 2
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
        ax1.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
            medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot([np.average(med.get_xdata())], [np.average(data[i])],
                 color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, numOfClasses + 0.5)
    top = 1.1
    bottom = 0.80
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(classes_names,
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(numOfClasses) + 1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(numOfClasses), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], top - (top * 0.05), upperLabels[tick],
                 horizontalalignment='center', size='x-small', weight=weights[k],
                 color=boxColors[k])

    # # Finally, add a basic legend
    # fig.text(0.80, 0.08, str(10) + ' Random Numbers',
    #          backgroundcolor=boxColors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=boxColors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
             size='x-small')

    plt.show()


values = result.get_box_plot_data().values()
boxplot(values)
