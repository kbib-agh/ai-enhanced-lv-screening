import heapq

import numpy as np


def find_path_dijkstra(img, start, end, mask=None):
    way = np.zeros(img.shape + (2,), dtype=np.uint16)

    summed = np.ones(img.shape, dtype=np.float32) * np.prod(img.shape) * np.max(img)
    summed[start[0], start[1]] = 0

    priority_queue = []
    heapq.heappush(priority_queue, (summed[start[0], start[1]], start))

    while priority_queue:
        priority, pix = heapq.heappop(priority_queue)
        # print(priority)
        if isinstance(end, tuple):
            if pix[0] == end[0] and pix[1] == end[1]:
                break
        elif isinstance(end, list):
            if isinstance(mask, np.ndarray):
                if mask[pix[0], pix[1]] in end:
                    break
            else:
                if pix in end:
                    break
        else:
            print('error input')
        for i in range(-1, 2):
            for j in range(-1, 2):
                if pix[0] + i >= 0 and pix[0] + i < img.shape[0] and pix[1] + j >= 0 and pix[1] + j < img.shape[1]:
                    if i * i + j * j >= 1:
                        # print(pix[0]+i,pix[1]+j,pix[2]+k)
                        if img[pix[0] + i, pix[1] + j] + summed[pix[0], pix[1]] < summed[pix[0] + i, pix[1] + j]:
                            way[pix[0] + i, pix[1] + j, 0] = pix[0]
                            way[pix[0] + i, pix[1] + j, 1] = pix[1]
                            summed[pix[0] + i, pix[1] + j] = img[pix[0] + i, pix[1] + j] + summed[pix[0], pix[1]]
                            heapq.heappush(priority_queue, (summed[pix[0] + i, pix[1] + j], (pix[0] + i, pix[1] + j)))

    path = [pix]
    while tuple(path[-1]) != tuple(start):
        path.append((way[path[-1][0], path[-1][1], 0], way[path[-1][0], path[-1][1], 1]))

    return path[::-1]
