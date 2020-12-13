import sys
import heapq


def readGraph(input_file):
    with open(input_file, 'r') as f:
        raw = [[float(x) for x in s.split(',')] for s in f.read().splitlines()]
    N = int(raw[0][0])
    m = int(raw[1][0])
    s = int(raw[2][0])
    adj_list = [[] for foo in range(N)]
    for edge in raw[3:]:
        adj_list[int(edge[0])].append((int(edge[1]), edge[2]))
    return N, m, s, adj_list


def writeOutput(output_file, N, mst):
    with open(output_file, 'w') as f:
        # output primstra tree (just neighbors, no edge weights)
        for j in range(N):
            neighbors = []
            for node in mst[j]:
                neighbors.append(node[0])
            # sort for the autograder
            neighbors.sort()
            f.write(','.join([str(i) for i in neighbors]) + '\n')


def make_undirected_graph(N, adj_list):
    G = {}
    for u in range(N):
        G[u] = {}

    # move our stuff in
    for u in range(N):
        for v in adj_list[u]:
            G[u][v[0]] = v[1]
            G[v[0]][u] = v[1]
    # back to list
    adj_list = ['x' for x in range(N)]
    for u in range(N):
        neighbors = []
        for v in G[u].keys():
            neighbors.append((v, G[u][v]))
        adj_list[u] = list(set(neighbors))
    return adj_list


def Run(input_file, output_file):
    N, m, s, adj_list = readGraph(input_file)
    undirected_adj_list = make_undirected_graph(N, adj_list)
    SPdistances, SPtotal_weight = dijkstra(N, m, s, undirected_adj_list)
    mst = kruskal(N, m, undirected_adj_list)
    MSTdistances, MSTtotal_weight = dijkstra(N, m, s, mst)
    best_TWR = float('inf')
    best_MDR = float('inf')
    primstra_tree = None
    best_c = -1
    # the line below generates 100 evenly spaced c values on the interval [0,1]. distance and weight will be tested on
    # each one of them and the min(max(TWR, MDR)) among all trials will be recorded
    c = [0.0 + x*(1.0-0.0)/100 for x in range(100)]
    for myc in c:
        temp_dist, temp_weight, temp_tree = primstra(N, m, s, undirected_adj_list, myc)
        temp_TWR = temp_weight/MSTtotal_weight
        temp_MDR = -1
        for i in range(N):
            if i != s:
                if temp_dist[i]/SPdistances[i] > temp_MDR:
                    temp_MDR = temp_dist[i]/SPdistances[i]
        if max(temp_TWR, temp_MDR) < max(best_TWR, best_MDR):
            best_TWR = temp_TWR
            best_MDR = temp_MDR
            primstra_tree = temp_tree
            best_c = myc
    print(best_TWR)
    print(best_MDR)
    print(best_c)
    writeOutput(output_file, N, primstra_tree)


def primstra(N, m, s, adj_list, c):
    # c is a constant, representing the trade-off, or a proportion factor used to determine the priority of each of the
    # two competing objectives. When c is 0, this algorithm is exactly the same as Prim's algorithm for finding MST. When
    # c is 1, this algorithm is exactly the same as Dijkstra's algorithm for finding SPT. This algorithm will be called
    # on a set of C and the tree that produces the min(max(TWR, MDR)) will be selected as output.

    # initialize the optimizer array. It is similar to the pi array that stores temporary distances in regular Dijkstra.
    # Each element of the optimizer array stores the result of the best known value to the optimization function
    # c * (path length to j) + (edge length ij). Initially this value is infinity for all nodes. Since this optimizer
    # is not the distance from source to a node anymore, there need to be a separate array recording distances
    optimizer = [float('inf')]*N
    # initialize best known distance array similar to the one in Dijkstra's. This array stores the best known distances
    # found by this algorithm as it builds the tree. However, optimization is no longer with respect to this parameter.
    # an interesting observation I found while testing my starter code is that best_dist eventually stores all the
    # correct final distances, so there is no need to have a separate distance array
    best_dist = [float('inf')]*N
    # initialize the array of parents to be -1 for all nodes, meaning we don't know the parent yet
    parents = [-1]*N
    # initialize the array to represent whether a node has been checked to FALSE for all nodes, meaning we
    # haven't checked anything
    checked = [False]*N
    # set the optimizer and best_dist to 0 for the source node S, and set the parent of the source node to None/NULL
    optimizer[s] = 0
    best_dist[s] = 0
    parents[s] = None
    # initialize the priority queue to contain a key-value pair, with key of a node V representing the best known
    # temporary distance from node V to S (first element of the tuple is the key, as min_heap pops off according to
    # the first element in the tuple by default). The value in the queue represent the node name (node names correspond
    # to position in the array, so it's also a representation of index. Initialize to (-1,-1) as placeholder, as it will
    # be changed very soon for all nodes
    Q = [(-1, -1)]*N
    # optimizer is infinity for all nodes except for the source, which was changed earlier. The while loop assigns key
    # value pairs (optimizer[node_name], node_name) to each element in the queue, initializing the optimization function
    # to infinity for all nodes except for the source node

    i = 0
    while i < N:
        Q[i] = (optimizer[i], i)
        i += 1
    # make Q a heap by calling heapify, imported heapq package
    heapq.heapify(Q)

    # while the queue is not empty, the node with the best known temporary distance will be popped and stored in cur_pair
    while len(Q) > 0:
        cur_pair = heapq.heappop(Q)
        # since you cannot modify the content within a tuple, when you adjust for temporary distance as you go, the only
        # way is to add a new tuple into the queue. Thus, a given node could have many tuples with different keys (or
        # weights) in the queue. Since the queue is a priority queue, a newly added tuple for a given node V would have
        # a lower key value than all the earlier added tuples for the given node V. Therefore, newly added tuple for V
        # would be placed before all the earlier tuples in the queue for V, and will be pop before all the earlier tuples
        # in the queue are popped. Thus, when you pop from queue, you need to check if the node has already been explored.
        # If it hasn't been explored, then that (key, value) pair must be the best known distance, or the last added one.
        # Since it is now the smallest key value in the entire queue, pop it and add the corresponding values to output.
        # If the node was already explored, it means that another tuple corresponding to the same node V with a lower key
        # value was popped before you reached this element in the queue for the same node V. This element is not the best
        # distance for V and you should just discard it. To discard it, move on to the next smallest key value in queue
        # by popped again and checking whether that node is checked. Use a while loop to make sure all the pairs that
        # should be discarded are indeed discarded. The special cases are for handling when the Q is of length 1 and is
        # popped. When the length of the queue is 1, pop results in the queue being empty. While loop checked for if the
        # length of the queue is 0. If it is 0, there's no next node to pop, so break out of the while loop and check
        # if the node that was just popped have indeed been checked. If it has, there's nothing to do and terminate the
        # outer while loop. If you don't write the if after the while loop, it would go on and do operations to calculate
        # distances whether the node has been checked or not. You don't want this, so break out of outer while.
        while checked[cur_pair[1]]:
            if len(Q) == 0:
                break
            else:
                cur_pair = heapq.heappop(Q)
        if checked[cur_pair[1]]:
            break

        # label the checked status of the popped node to be True
        checked[cur_pair[1]] = True
        # check all vertices in the adjacency list of the newly added vertex
        for v in adj_list[cur_pair[1]]:
            # if it hasn't been checked, meaning if it doesn't belong to the set of connected component
            # then it is eligible under prim's (or that only edges with one end in the connected set should be considered
            if checked[v[0]] is False:
            # for nodes that are qualified under prim's criterion, check if the best known value for their optimization
            # function is greater than this newly discovered potential optimization function value. If it is better, change
            # that value to be the newly discovered optimization function value. Set parent index, and push the new tuple
            # onto the queue. Also, check whether using this path produces a better distance. If it does, update distance
            # to use this given edge as well
                if optimizer[v[0]] > c*best_dist[cur_pair[1]] + v[1]:
                    optimizer[v[0]] = c*best_dist[cur_pair[1]] + v[1]
                    parents[v[0]] = cur_pair[1]
                    heapq.heappush(Q, (optimizer[v[0]], v[0]))
                    if best_dist[v[0]] > best_dist[cur_pair[1]] + v[1]:
                        best_dist[v[0]] = best_dist[cur_pair[1]] + v[1]

    # best_dist contains the distances of all nodes from the source node. The max value in best_dist is the maximum
    # distance from s to a node in this given tree among all nodes. This will be returned to compare against maximum
    # distance in SPT to compute MDR

    # construct a dictionary. Add (u, v) and (v, u) pairs as keys in the dictionary for each edge in the adjacency list.
    # Since the adjacency list is undirected, it need to be added this way. Initialize total weight to be 0, and for
    # each node that's not the source, key the dictionary by (node, parent[node]) to find the weight of the edge
    # given by the tree. Since we don't know which key value pair (u, v), or (v, u) will be selected, adding both to
    # the dictionary guarantees that one of them will be there. Since each node has exactly one parent, the loop goes
    # through the list exactly once to find total weight while avoiding double-counting.
    # the tree need to be outputted. Throughout the loop, add edges to tree_element. The resultant tree_element would
    # contain each edge exactly once. Since the output is for an undirected graph, adjust later to compensate
    mydict = {}
    tree_element = []
    for j in range(N):
        for k in range(len(adj_list[j])):
            if ((j, adj_list[j][k][0]) not in mydict) and ((adj_list[j][k][0], j) not in mydict):
                mydict[(j, adj_list[j][k][0])] = adj_list[j][k][1]
                mydict[(adj_list[j][k][0]), j] = adj_list[j][k][1]
    total_weight = 0
    for i1 in range(N):
        if i1 != s:
            total_weight += mydict[(i1, parents[i1])]
            # append to tree_element in the format (weight, start node, end node)
            tree_element.append((mydict[(i1, parents[i1])], parents[i1], i1))

    # from tree_element, loop through each element and add each edge two times to created undirected output
    optimal_tree_list = [list() for x in range(N)]
    for tree_elem in tree_element:
        optimal_tree_list[tree_elem[1]].append((tree_elem[2], tree_elem[0]))
        optimal_tree_list[tree_elem[2]].append((tree_elem[1], tree_elem[0]))

    return best_dist, total_weight, optimal_tree_list


# my modified Dijkstra's algorithm from part 2
def dijkstra(N, m, s, adj_list):
    tempdist = [float('inf')]*N
    distances = [float('inf')]*N
    parents = [-1]*N
    checked = [False]*N
    tempdist[s] = 0
    parents[s] = None
    Q = [(-1, -1)]*N
    i = 0
    while i < N:
        Q[i] = (tempdist[i], i)
        i += 1
    heapq.heapify(Q)

    while len(Q) > 0:
        cur_pair = heapq.heappop(Q)
        while checked[cur_pair[1]]:
            if len(Q) == 0:
                break
            else:
                cur_pair = heapq.heappop(Q)
        if checked[cur_pair[1]]:
            break
        checked[cur_pair[1]] = True
        distances[cur_pair[1]] = tempdist[cur_pair[1]]
        for v in adj_list[cur_pair[1]]:
            if tempdist[v[0]] > tempdist[cur_pair[1]] + v[1]:
                tempdist[v[0]] = tempdist[cur_pair[1]] + v[1]
                parents[v[0]] = cur_pair[1]
                heapq.heappush(Q, (tempdist[v[0]], v[0]))

    dict = {}
    for j in range(N):
        for k in range(len(adj_list[j])):
            if ((j, adj_list[j][k][0]) not in dict) and ((adj_list[j][k][0], j) not in dict):
                dict[(j, adj_list[j][k][0])] = adj_list[j][k][1]
                dict[(adj_list[j][k][0]), j] = adj_list[j][k][1]
    total_weight = 0
    for i1 in range(N):
        if i1 != s:
            total_weight += dict[(i1, parents[i1])]
    return distances, total_weight


# my modified Kruskal's algorithm from part 2 of the assignment
def kruskal(N, m, undirected_adj_list):
    dict = {}
    edge = []
    for j in range(N):
        for k in range(len(undirected_adj_list[j])):
            if ((j, undirected_adj_list[j][k][0]) not in dict) and ((undirected_adj_list[j][k][0], j) not in dict):
                edge.append((undirected_adj_list[j][k][1], j, undirected_adj_list[j][k][0]))
                dict[(j, undirected_adj_list[j][k][0])] = "placeholder"
    edge = sorted(edge)

    T = []
    head = [None]*N
    size = [None]*N
    children = [set() for x in range(N)]

    # make union here, all nodes is their own heads, and all islands have size 1
    for j1 in range(N):
        head[j1] = j1
        size[j1] = 1
        children[j1].add(j1)

    for elem in edge:
        if head[elem[1]] != head[elem[2]]:
            T.append(elem)
            if size[head[elem[1]]] <= size[head[elem[2]]]:
                size[head[elem[2]]] += size[head[elem[1]]]
                for mynode in children[head[elem[1]]]:
                    head[mynode] = head[elem[2]]
                    children[head[elem[2]]].add(mynode)
            else:
                size[head[elem[1]]] += size[head[elem[2]]]
                for mynode in children[head[elem[2]]]:
                    head[mynode] = head[elem[1]]
                    children[head[elem[1]]].add(mynode)

    mst_adj_list = [list() for x in range(N)]
    for final in T:
        mst_adj_list[final[1]].append((final[2], final[0]))
        mst_adj_list[final[2]].append((final[1], final[0]))

    return mst_adj_list


def main(args=[]):
    Run('g_randomEdges.txt', 'outputrandom')
    Run('g_donutEdges.txt', 'outputdonut')
    Run('g_zigzagEdges.txt', 'outputzigzag')

if __name__ == "__main__":
    main(sys.argv[1:])
