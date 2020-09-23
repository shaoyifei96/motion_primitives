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
        mp = self.edges[i, j]
        if mp is None:
            return None
        explored = []
        queue = [[mp]]
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
                        if sum([m.cost for m in new_path]) < 2*self.dispersion:
                            queue.append(new_path)
                            # return path if neighbor is goal
                            if(neighbor.start_state == mp.end_state).all():
                                paths.append(new_path)

                # mark node as explored
                explored.append(node)
        for path in paths:
            for mp in path:
                print(mp.start_state)
                print(mp.end_state)
            print('end')
        return paths


if __name__ == "__main__":
    mpl = MotionPrimitiveLattice.load("lattice_test.json")
    mpl.reduce_graph_degree()
