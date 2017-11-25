#!/usr/bin/env python3

# Wishlist:
# 1. disjoint set operations
# 2. dijkstra
# 3. splay tree operations
# 4. all intersections

# Solving shortlist:
# 1. Dynamic programming (lru_cache)
# 2. Binary search on decision problem

def maxflow(edges, source, sink):
    '''Dinic' algorithm, O(V^2 * E)
    >>> edges = {1: {2: 3, 3: 3}, 2: {3: 2, 4: 3}, 3: {5: 2},
    ...          4: {5: 4, 6: 2}, 5: {6: 3}, 6: {}}
    >>> maxflow(edges, 1, 6)
    5
    >>> for i in range(1, 7):
    ...     print(*[edges[i].get(j, '-') for j in range(1, 7)])
    - 0 1 - - -
    3 - 2 0 - -
    2 0 - - 0 -
    - 3 - - 3 0
    - - 2 1 - 0
    - - - 2 3 -
    '''
    INF = 2 ** 64
    def dfs(x, inflow):
        if x == source:
            return inflow
        rem = inflow
        for y in edges[x]:
            cap = edges[y][x]
            if cap > 0 and level[x] > level.get(y, INF):
                used = dfs(y, min(cap, rem))
                edges[x][y] += used
                edges[y][x] -= used
                rem -= used
                if used < min(cap, rem):
                    del level[y]
                if rem == 0:
                    return inflow
        return inflow - rem
    for x, ys in edges.items():
        for y in ys:
            edges[y].setdefault(x, 0)
    while True:
        level = {source: 0}
        n = 1
        todo = [source]
        while todo:
            if sink in todo:
                break
            newtodo = []
            for x in todo:
                for y, cap in edges[x].items():
                    if cap > 0 and y not in level:
                        level[y] = n
                        newtodo.append(y)
            n += 1
            todo = newtodo
        if sink not in level:
            return sum(edges[sink].values())
        dfs(sink, INF)

def convexhull(points):
    '''Andrew's monotone chain, O(n log n)
    >>> convexhull([(1, 1, 'a'), (0, -3, 'b'), (-1, 1, 'c'), (-1, -1, 'd'),
    ...             (3, 0, 'e'), (-3, 0, 'f'), (1, -1, 'g'), (0, 3, 'h')])
    [(-3, 0, 'f'), (0, 3, 'h'), (3, 0, 'e'), (0, -3, 'b')]
    >>> convexhull([(1, 1), (-1, 1), (1, -1), (0, 0), (-1, -1)])
    [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    '''
    def half(points):
        res = []
        for p in points:
            while len(res) > 1 and \
                    (p[1] - res[-1][1]) * (res[-1][0] - res[-2][0]) > \
                    (p[0] - res[-1][0]) * (res[-1][1] - res[-2][1]):
                res.pop()
            res.append(p)
        res.pop()
        return res
    points = sorted(points)
    return half(points) + half(reversed(points))

def dfs(edges, todo):
    '''
    >>> edges = {1: [2, 6], 2: [3, 5], 3: [4], 4: [2, 8],
    ...          5: [4, 7], 6: [5, 7], 7: [6], 8: [7]}
    >>> list(dfs(edges, [1]))
    [1, 2, 3, 4, 8, 7, 6, 5]
    >>> list(dfs({1: [], 2: [1]}, [1]))
    [1]
    '''
    done = set()
    while todo:
        x = todo.pop()
        if x not in done:
            done.add(x)
            yield x
            todo.extend(reversed(edges[x]))

def bfs(edges, todo, finish=[]):
    '''
    >>> edges = {1: [2, 6], 2: [3, 5], 3: [4], 4: [2, 8],
    ...          5: [4, 7], 6: [5, 7], 7: [6], 8: [7]}
    >>> list(bfs(edges, [1]))
    [[1], [2, 6], [3, 5, 7], [4], [8]]
    >>> list(bfs(edges, [1], [3]))
    [[1], [2, 6], [3, 5, 7]]
    >>> list(bfs({1: [], 2: [1]}, [1]))
    [[1]]
    '''
    done = set()
    while todo:
        yield todo
        if any(x in finish for x in todo):
            return
        newtodo = []
        for x in todo:
            for y in edges[x]:
                if y not in done:
                    done.add(y)
                    newtodo.append(y)
        todo = newtodo

def primelist(n):
    '''Sieve of Eratosthenes, O(n log n)
    >>> primelist(23)
    [2, 3, 5, 7, 11, 13, 17, 19]
    '''
    if n < 3:
        return []
    pl = [True] * (n // 2)
    for k in range(3, int(n ** 0.5) + 1, 2):
        if pl[k // 2]:
            kk = (k * k) // 2
            pl[kk::k] = [False] * len(range(kk, n // 2, k))
    r = [i for i, x in zip(range(1, n, 2), pl) if x]
    r[0] = 2
    return r

def minspantree(edges, start):
    '''Prim's algorithm, O(E log V)
    >>> edges = {1: {2: 3, 4: 1}, 2: {1: 3, 4: 2},
    ...          3: {4: 4}, 4: {1: 1, 2: 2, 3: 4}}
    >>> minspantree(edges, next(iter(edges)))
    {1: {4: 1}, 4: {1: 1, 2: 2, 3: 4}, 2: {4: 2}, 3: {4: 4}}
    '''
    from heapq import heappush, heappop, heapify
    mst = {start: dict()}
    frontier = [(d, n, start) for n, d in edges[start].items()]
    heapify(frontier)
    while frontier:
        d, v, p = heappop(frontier)
        if v in mst:
            continue
        mst[v] = {p: d}
        mst[p][v] = d
        for n, d in edges[v].items():
            heappush(frontier, (d, n, v))
    return mst

def minspantree2(dists):
    '''Prim's algorithm on adjacency matrices, O(V^2)
    >>> dists = [[0, 3, 7, 1], [3, 0, 7, 2], [7, 7, 0, 4], [1, 2, 4, 0]]
    >>> minspantree2(dists)
    {0: {3: 1}, 3: {0: 1, 1: 2, 2: 4}, 1: {3: 2}, 2: {3: 4}}
    '''
    mst = {0: dict()}
    todo = [(d, t, 0) for t, d in enumerate(dists[0]) if t != 0]
    while todo:
        d, t, s = min(todo)
        mst[t] = {s: d}
        mst[s][t] = d
        todo = [(od, ot, os) if od < dists[ot][t] else (dists[ot][t], ot, t)
                for od, ot, os in todo if t != ot]
    return mst

# LINEAR ALGEBRA

def param(x1, y1, x2, y2, x, y):
    '''Value of projection of x, y on line if x1, y1 is 0 and x2, y2 is 1
    >>> param(1, 1, 2, 3, 3.4, 0.8)
    0.4
    '''
    d1x, d1y = x2 - x1, y2 - y1                                 # d1 = p2 - p1
    d2x, d2y = x - x1, y - y1                                   # d2 = p - p1
    return (d1x * d2x + d1y * d2y) / (d1x * d1x + d1y * d1y)    # d1.d2 / d1.d1

def project(x1, y1, x2, y2, x, y):
    '''Project point x, y on line defined by x1, y1 and x2, y2
    >>> project(1, 1, 2, 3, 3.4, 0.8)
    (1.4, 1.8)
    '''
    p = param(x1, y1, x2, y2, x, y)
    return x1 + p * (x2 - x1), y1 + p * (y2 - y1)

def mirror(x1, y1, x2, y2, x, y):
    '''Mirror point x, y over line defined by x1, y1 and x2, y2
    >>> mirror(1, 1, 3, 2, 2, 4)
    (4.0, 0.0)
    '''
    x3, y3 = project(x1, y1, x2, y2, x, y)  # p3 = project(p1, p2, p)
    return x3 * 2 - x, y3 * 2 - y           # p3 * 2 - p

def dist(x1, y1, x2, y2):
    '''Distance from point x1, y1 to point x2, y2
    >>> dist(1, 1, 4, 5)
    5.0
    '''
    from math import sqrt
    dx, dy = x2 - x1, y2 - y1
    return sqrt(dx * dx + dy * dy)

def linepointdist(x1, x2, y1, y2, x, y):
    '''Distance from point x, y to line defined by x1, y1 and x2, y2
    >>> linepointdist(1, 1, 4, 5, -4, 4)
    5.0
    '''
    x3, y3 = project(x1, y1, x2, y2, x, y)
    return dist(x3, y3, x, y)

def intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    '''Intersection point between two lines
    >>> intersection(1, 1, 5, 3, 2, 3, 5, 0)
    (3.0, 2.0)
    '''
    d1x, d1y = x1 - x2, y1 - y2
    d2x, d2y = x3 - x4, y3 - y4
    D = d1x * d2y - d1y * d2x
    if not D:
        return None
    p1, p2 = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
    return (p1 * d2x - d1x * p2) / D, (p1 * d2y - d1y * p2) / D

def hasintersection(x1, y1, x2, y2, x3, y3, x4, y4):
    '''Wether the two given line segments intersect
    >>> hasintersection(1, 1, 2, 3, 2, 1, 1, 4)
    True
    '''
    x, y = intersection(x1, y1, x2, y2, x3, y3, x4, y4)
    p1 = param(x1, y1, x2, y2, x, y)
    p2 = param(x3, y3, x4, y4, x, y)
    return 0 <= p1 <= 1 and 0 <= p2 <= 1

if __name__ == '__main__':
    import doctest
    doctest.testmod()
