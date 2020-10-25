import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os


def change_points():
    sky_size = [1080, 1080]

    cmb_mu, cmb_sigma = 2.7, 0.2
    # cmb = np.random.normal(cmb_mu, cmb_sigma, sky_size[0] * sky_size[1])
    # cmb = [x % 10 + (x % 100) / 10 for x in range(sky_size[0] * sky_size[1])]
    # cmb = np.resize(cmb, sky_size)

    tile_size = 10
    tile1 = np.concatenate([[[0] * tile_size] * tile_size, [[1] * tile_size] * tile_size], 0)
    tile2 = np.concatenate([[[1] * tile_size] * tile_size, [[0] * tile_size] * tile_size], 0)
    tile = np.concatenate([tile1, tile2], 1)
    cmb = np.tile(tile, [int(sky_size[0] / (2*tile_size)), int(sky_size[1] / (2*tile_size))])


    img = Image.fromarray(cmb * 255 / np.amax(cmb))
    img.show()

    eta_mu, eta_sigma = 0, 0.3
    x, y = np.meshgrid(np.linspace(-1, 1, sky_size[1]), np.linspace(-1, 1, sky_size[0]))
    d = np.sqrt(x * x + y * y)
    eta = 10000 * np.exp(-((d - eta_mu) ** 2 / (2.0 * eta_sigma ** 2)))

    img = Image.fromarray(abs(eta * 255) / np.amax(abs(eta)))
    img.show()

    sky_grid = np.empty([sky_size[0], sky_size[1], 2])

    for i in range(0, sky_size[0]):
        for j in range(0, sky_size[1]):
            sky_grid[i, j] = [i, j]

    for (x, y), element in np.ndenumerate(eta):
        sky_grid[x, y] += [element, element]
        sky_grid[(x + 1) % sky_size[0], y] += [-element, element]
        sky_grid[x, (y + 1) % sky_size[1]] += [element, -element]
        sky_grid[(x + 1) % sky_size[0], (y + 1) % sky_size[1]] += [-element, -element]

    sky_grid = np.round(sky_grid).astype(int)
    for i in range(0, sky_size[0]):
        for j in range(0, sky_size[1]):
            sky_grid[i, j][0] %= sky_size[0]
            sky_grid[i, j][1] %= sky_size[1]

    print(sky_grid[0, 0])

    cmb_ren = np.empty(sky_size)
    for x in range(sky_grid.shape[0]):
        for y in range(sky_grid.shape[1]):
            cmb_ren[x, y] = cmb[sky_grid[x, y][0], sky_grid[x, y][1]]

    img = Image.fromarray(cmb_ren * 255 / np.amax(cmb))
    img.show()

    cmb_diff = abs(cmb_ren - cmb)

    img = Image.fromarray(cmb_diff * 255 / np.amax(cmb_diff))
    img.show()

    points = 0

    return points
