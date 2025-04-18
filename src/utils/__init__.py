import numpy
import matplotlib

matplotlib.use("AGG")
from matplotlib import pyplot


def plot_stroke(stroke, save_name=None):
    # helps with getting it to show up early in training
    stroke = numpy.append(stroke, [[1, 0, 0]], axis=0)

    print(stroke)

    # Plot a single example.
    f, ax = pyplot.subplots()

    x = numpy.cumsum(stroke[:, 1])
    y = numpy.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.0
    size_y = y.max() - y.min() + 1.0

    f.set_size_inches(5.0 * size_x / size_y, 5.0)

    cuts = numpy.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value], "k-", linewidth=3)
        start = cut_value + 1

    ax.axis("off")  # equal
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(save_name, bbox_inches="tight", pad_inches=0.5)
        except Exception as e:
            print("Error building image!: " + save_name)
            print(e)

    pyplot.close()