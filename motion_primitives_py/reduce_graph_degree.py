from motion_primitives_py import MotionPrimitiveLattice


class MotionPrimitiveLattice(MotionPrimitiveLattice):
    def reduce_graph_degree(self):
        print(len(self.vertices))
        print(len(self.edges))
        for i in range(len(self.edges)):
            for j in range(len(self.vertices)):
                if i != j:
                    self.bfs(i, j)

    def bfs(self, i, j):
        # bfs https://pythoninwonderland.wordpress.com/2017/03/18/how-to-implement-breadth-first-search-in-python/
        original_edge = self.edges[i, j]
        if original_edge is None:
            return None
        explored = []
        queue = [[original_edge]]
        paths = []
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
                                if new_path_cost > original_edge.cost:
                                    self.edges[i, j] = None

                # mark node as explored
                explored.append(node)

        # for path, cost in paths:
        #     print(2*self.dispersion)
        #     print(original_edge.cost)
        #     print(cost)
        # print(original_edge.start_state)
        # print(original_edge.end_state)

        # for mp in path:
        #     print(mp.start_state)
        #     print(mp.end_state)
        # return paths


if __name__ == "__main__":
    import numpy as np

    mpl = MotionPrimitiveLattice.load("lattice_test.json")
    print(sum([1 for i in np.nditer(mpl.edges, ['refs_ok']) if i != None])/len(mpl.vertices))

    mpl.reduce_graph_degree()
    print(sum([1 for i in np.nditer(mpl.edges, ['refs_ok']) if i != None])/len(mpl.vertices))
