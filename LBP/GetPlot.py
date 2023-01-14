import matplotlib.pyplot as plt
import json
import numpy as np

# Function for plotting data of every individual LBP method (I used same parameters for all of them)
def plotThis(data, title, minimum, maximum):
    euc = data['euclidean']
    cos = data['cosine']

    # Preparing data of the resutls calculated with euclidean distance
    xEuc, yEuc, zEuc = [], [], []
    for radKey in euc:
        samples = euc[radKey]
        for samplKey in samples:
            value = samples[samplKey]
            xEuc.append(int(radKey))
            yEuc.append(int(samplKey))
            zEuc.append(value)

    xEuc = np.array(xEuc)
    yEuc = np.array(yEuc)
    zEuc = np.array(zEuc)

    # Preparing data of the resutls calculated with cosine distance
    xCos, yCos, zCos = [], [], []
    for radKey in cos:
        samples = cos[radKey]
        for samplKey in samples:
            value = samples[samplKey]
            xCos.append(int(radKey))
            yCos.append(int(samplKey))
            zCos.append(value)

    xCos = np.array(xCos)
    yCos = np.array(yCos)
    zCos = np.array(zCos)

    # Shifting and normalizing data for a clearer graph
    # I shift all data for 0*95 of the minimum value towards the zero, so the differences in results (bubbles)
    # is better visible. 
    zAll = np.stack((zCos, zEuc))
    zAll = zAll - minimum*0.95
    zAll = zAll / maximum
    
    # ploting the data for cosine distance
    plt.figure(figsize=(12, 8))
    plt.scatter(xCos-0.2, yCos-0.2, 
                    color='green', 
                    alpha=0.5,
                    s = zAll[0] * 1000,
                    label = "Cosine distance",
                    )
    # and also for the euclidean
    plt.scatter(xEuc+0.2, yEuc+0.2, 
                    color='red', 
                    alpha=0.5,
                    s = zAll[1] * 1000,
                    label = "Euclidean distance"
                    )

    # adding labels, titles and legend... making it pretty :)
    for i, txt in enumerate(zEuc):
        plt.annotate(str(txt)[0:5]+"%", (xEuc[i]+0.2, yEuc[i]+0.2))
    for i, txt in enumerate(zCos):
        plt.annotate(str(txt)[0:5]+"%", (xCos[i]-0.2, yCos[i]-0.2))
    plt.xlabel("Radius", size=14)
    plt.xticks(np.arange(xCos.min(), xCos.max()+1, 2))
    plt.ylabel("Samples", size=14)
    plt.yticks(np.arange(yCos.min(), yCos.max()+1, 2))
    plt.title(title, size=20, pad=25)
    plt.legend(bbox_to_anchor =(1, 1.15), markerscale=0.5, title="Distance method used", fancybox=True)
    plt.show()


# the getPlot() function uses the above plotThis() function to make plots for all 3 different 
# types of LBP that I use.

def getPlot():
    with open('Results/final.json', 'r') as file:
        data = json.load(file)

        # getting the minimum and maximum of ALL results for appropriate buuble scaling on the graphs.
        results = []
        for key1 in data:
            methods = data[key1]
            for key2 in methods:
                radii = methods[key2]
                for key3 in radii:
                    samples=radii[key3]
                    for value in samples.values():
                        results.append(value)

        results = np.array(results)

        for key in data:
            plotThis(data[key], "Method used: "+key, results.min(), results.max())

getPlot()