from motion_primitives_py import MotionPrimitiveLattice


class MotionPrimitiveLattice(MotionPrimitiveLattice):
    def reduce_graph_degree(self):
        pts, independent = self.uniform_state_set(self.max_state[:self.control_space_q], self.resolution[:self.control_space_q], random=False)

        print(len(self.vertices))
        print(len(self.edges))
        paths = np.empty((len(self.edges), len(self.vertices)), dtype=object)
        for i in range(len(self.edges)):
            for j in range(len(self.vertices)):
                if i != j:
                    paths[i, j] = self.bfs(i, j)

        counter = 0
        for i in range(len(self.edges)):
            for j in range(len(self.vertices)):
                if paths[i, j] is not None:
                    if len(paths[i, j]) > 1:
                        counter +=1
                        candidate_path = np.argmax([cost for path, cost in paths[i, j]])
                        # for k, (path, cost) in enumerate(paths[i,j]):
                            # if k!= candidate_path:
                                # for edge in path:
                                    # print(independent)
                                    # print(np.where(independent==np.array(edge.start_state)))
                                    # print(edge.start_state)
                                    # print(edge.end_state)
        print(counter)
        print(sum([1 for i in np.nditer(mpl.edges, ['refs_ok']) if i != None]))
        
    def bfs(self, i, j):
        # bfs https://pythoninwonderland.wordpress.com/2017/03/18/how-to-implement-breadth-first-search-in-python/
        original_edge = self.edges[i, j]
        if original_edge is None:
            return None
        explored = []
        queue = [[original_edge]]
        paths = [([original_edge], original_edge.cost)]
        while queue:
            # pop the first path from the queue
            path = queue.pop(0)
            # get the last node from the path
            node = path[-1]
            if node not in explored:
                # neighbors = self.edges[:, node]
                neighbors = self.get_neighbor_mps(j)
                # go through all neighbor nodes, construct a new path and
                # push it into the queue
                for neighbor in neighbors:
                    if neighbor is not None and not (neighbor.start_state == neighbor.end_state).all():
                        new_path = list(path)
                        new_path.append(neighbor)
                        # print([(x.start_state, x.end_state)for x in new_path])
                        # print(mp.end_state)
                        # print(mp.cost)
                        # print(2*self.dispersion)
                        # print(sum([m.cost for m in new_path]))
                        new_path_cost = sum([m.cost for m in new_path])
                        if new_path_cost < 2*self.dispersion:
                            queue.append(new_path)
                            # return path if neighbor is goal
                            if(neighbor.end_state == original_edge.end_state).all():
                                paths.append((new_path, new_path_cost))

                # mark node as explored
                explored.append(node)
        return paths


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    mpl = MotionPrimitiveLattice.load("lattice_test.json", True)
    print(sum([1 for i in np.nditer(mpl.edges, ['refs_ok']) if i != None])/len(mpl.vertices))

    mpl.reduce_graph_degree()
    print(sum([1 for i in np.nditer(mpl.edges, ['refs_ok']) if i != None])/len(mpl.vertices))
    mpl.limit_connections(np.inf)
    plt.show()
