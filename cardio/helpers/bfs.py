import heapq

import numpy as np


def find_path_bfs(img, start, end):
    priority_queue = []
    visited = np.zeros_like(img).astype(bool)
    track = np.zeros(img.shape + (2,), dtype=np.int16)

    heapq.heappush(priority_queue, (img[start[0], start[1]], start))
    visited[start[0], start[1]] = 1

    neighbors = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if i * i + j * j >= 1]

    step = 0
    while priority_queue:

        # if step%100000 == 0:
        #     print(step,len(priority_queue),np.prod(img.shape))

        priority, pix = heapq.heappop(priority_queue)
        if isinstance(end, tuple):
            if pix == end:
                break
        elif isinstance(end, list):
            if pix in end:
                break
        else:
            raise Exception('BFS input error')

        for i, j in neighbors:
            new_r, new_c = pix[0] + i, pix[1] + j

            if 0 <= new_r < img.shape[0] and 0 <= new_c < img.shape[1] and not visited[new_r, new_c]:
                visited[new_r, new_c] = True
                track[new_r, new_c] = pix
                heapq.heappush(priority_queue, (img[new_r, new_c], (new_r, new_c)))
        step += 1

    path = [pix]
    while tuple(path[-1]) != tuple(start):
        path.append(tuple(track[path[-1]]))

    return path[::-1]
