import numpy
import torch
import matplotlib

matplotlib.use("AGG")
from matplotlib import pyplot
from matplotlib.patches import Ellipse


def plot_stroke(stroke, save_name=None):
    # helps with getting it to show up early in training
    stroke = numpy.append(stroke, [[1, 0, 0]], axis=0)

    # for point in stroke:
    #     print(point)

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


def visualize_mdn_overlay(stroke, y_hat, save_name, top_k=5, scale=1.0):
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    # offsets -> absolute coordinates
    xy = np.cumsum(stroke[:, 1:], axis=0)
    x  = xy[:, 0]
    y  = xy[:, 1]

    fig, ax = plt.subplots()
    ax.plot(x, -y, lw=1, color="black")

    split_sizes = [1] + [20] * 6
    parts = torch.split(y_hat.squeeze(0), split_sizes, dim=1)

    pi      = torch.softmax(parts[1], dim=1).cpu().numpy()   # (T,20)
    mu_1    = parts[2].cpu().numpy()
    mu_2    = parts[3].cpu().numpy()
    std_1   = torch.exp(parts[4]).cpu().numpy()
    std_2   = torch.exp(parts[5]).cpu().numpy()
    rho     = torch.tanh(parts[6]).cpu().numpy()

    for t in range(stroke.shape[0]):
        cx, cy = x[t], y[t]
        best   = np.argsort(-pi[t])[:top_k]

        for k in best:
            w  = pi[t, k]
            mx = cx + mu_1[t, k]
            my = cy + mu_2[t, k]

            # draw mean
            ax.scatter(mx, -my, s=15, c="red", alpha=0.25 * w)

            # covariance -> ellipse
            s1 = std_1[t, k] * scale
            s2 = std_2[t, k] * scale
            r  = rho[t, k]

            # convert rho to ellipse angle (radians -> degrees)
            angle = 0.5 * np.arctan2(2 * r * s1 * s2, s1**2 - s2**2)
            angle_deg = np.degrees(angle)

            # eigenvalues (axes lengths squared)
            lam1 = 0.5 * (s1**2 + s2**2 +
                          np.sqrt((s1**2 - s2**2)**2 + 4 * r**2 * s1**2 * s2**2))
            lam2 = 0.5 * (s1**2 + s2**2 -
                          np.sqrt((s1**2 - s2**2)**2 + 4 * r**2 * s1**2 * s2**2))

            width  = 2 * np.sqrt(lam1)
            height = 2 * np.sqrt(lam2)

            e = Ellipse((mx, -my),
                        width=width,
                        height=height,
                        angle=-angle_deg,          # negate because of y flip
                        linewidth=0,
                        facecolor="red",
                        alpha=0.15 * w)
            ax.add_patch(e)

    ax.axis("equal")
    ax.axis("off")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()