import numpy as np
from matplotlib import pyplot as plt
import glob


def draw(graph_name: str, config: list):
    # print(data)
    fig, ax = plt.subplots(1, 3)
    for i in config:

        filename = graph_name+"."+".".join(i)+".stats"
        acc = graph_name+"."+".".join(i)+".accu"
        acc = np.loadtxt(acc)
        data = np.loadtxt(filename)

        print("{} {} {} {}".format(graph_name, ".".join(
            i), str(acc), data[-1][0]/data[-1][1]))


if __name__ == "__main__":
    all_files = glob.glob("stats/*/*.stats")
    print(all_files)
    exit()
    name_split = [i.split(".")[0:-1] for i in all_files]
    graph_to_names = dict()

    graph_to_names["cora"] = []
    graph_to_names["citeseer"] = []
    graph_to_names["pubmed"] = []

    for i in name_split:
        graph_to_names[i[0]].append(i[1:])

    for i in graph_to_names:
        print(i)
        draw(i, graph_to_names[i])
