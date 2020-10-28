import numpy as np
from PIL import Image
import math


def change_points(render_method='interp', verbose=0):
    # --------- TEST INPUT BEGIN
    sky_size = [1000, 1000]
    verbose = 1
    # cmb_mu, cmb_sigma = 2.7, 0.2
    # cmb = np.random.normal(cmb_mu, cmb_sigma, sky_size[0] * sky_size[1])
    # cmb = [x % 10 + (x % 100) / 10 for x in range(sky_size[0] * sky_size[1])]
    # cmb = np.resize(cmb, sky_size)

    tile_size = 10
    tile1 = np.concatenate([[[0] * tile_size] * tile_size, [[1] * tile_size] * tile_size], 0)
    tile2 = np.concatenate([[[1] * tile_size] * tile_size, [[0] * tile_size] * tile_size], 0)
    tile = np.concatenate([tile1, tile2], 1)
    cmb = np.tile(tile, [int(sky_size[0] / (2 * tile_size)), int(sky_size[1] / (2 * tile_size))])

    img = Image.fromarray(cmb * 255 / np.amax(cmb))
    img.show()

    eta_mu, eta_sigma = 0, 0.3
    x, y = np.meshgrid(np.linspace(-1, 1, sky_size[1]), np.linspace(-1, 1, sky_size[0]))
    d = np.sqrt(x * x + y * y)
    eta = 10000 * np.exp(-((d - eta_mu) ** 2 / (2.0 * eta_sigma ** 2)))
    # eta = 100000 * np.ones(sky_size)
    # --------- TEST INPUT END

    # Display magnification Image
    if verbose >= 1:
        img = Image.fromarray(abs(eta * 255) / np.amax(abs(eta)))
        img.show()

    # Create map from cmb to cmb_ren; this map exists as sky_map
    sky_grid = np.empty([sky_size[0], sky_size[1], 2])
    for (x, y), element in np.ndenumerate(eta):
        sky_grid[x, y] += [x, y]
        sky_grid[x, y] += [element, element]
        sky_grid[(x + 1) % sky_size[0], y] += [-element, element]
        sky_grid[x, (y + 1) % sky_size[1]] += [element, -element]
        sky_grid[(x + 1) % sky_size[0], (y + 1) % sky_size[1]] += [-element, -element]

    cmb_ren = np.empty(sky_size)

    # 2-D Interpolation Rendering Method
    if render_method == 'interp':
        sky_grid = np.round(sky_grid, 4)
        for x in range(0, sky_size[0]):
            for y in range(0, sky_size[1]):
                sky_grid[x, y][0] %= sky_size[0]
                sky_grid[x, y][1] %= sky_size[1]
                interp1 = cmb[math.floor(sky_grid[x, y][0]), math.floor(sky_grid[x, y][1])]
                interp2 = cmb[math.ceil(sky_grid[x, y][0]), math.floor(sky_grid[x, y][1])]
                interp3 = cmb[math.floor(sky_grid[x, y][0]), math.ceil(sky_grid[x, y][1])]
                interp4 = cmb[math.ceil(sky_grid[x, y][0]), math.ceil(sky_grid[x, y][1])]
                interp1_2 = interp1 * (1 - (sky_grid[x, y][0] % 1)) + interp2 * (sky_grid[x, y][0] % 1)
                interp3_4 = interp3 * (1 - (sky_grid[x, y][0] % 1)) + interp4 * (sky_grid[x, y][0] % 1)
                cmb_ren[x, y] = interp1_2 * (1 - (sky_grid[x, y][1] % 1)) + interp3_4 * (sky_grid[x, y][1] % 1)

    # Nearest Neighbor Rendering Method
    if render_method == 'nearest':
        sky_grid = np.round(sky_grid).astype(int)
        for x in range(sky_grid.shape[0]):
            for y in range(sky_grid.shape[1]):
                sky_grid[x, y][0] %= sky_size[0]
                sky_grid[x, y][1] %= sky_size[1]
                cmb_ren[x, y] = cmb[sky_grid[x, y][0], sky_grid[x, y][1]]

    # Display rendered cmb image
    if verbose >= 1:
        img = Image.fromarray(cmb_ren * 255 / np.amax(cmb))
        img.show()

    # Calculate and display difference between original cmb and rendered cmb
    if verbose >= 1:
        cmb_diff = abs(cmb_ren - cmb)
        img = Image.fromarray(cmb_diff * 255 / np.amax(cmb_diff))
        img.show()

    return cmb_ren
