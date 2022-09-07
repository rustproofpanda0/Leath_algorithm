import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors


class PercCluster():
    def __init__(self, size, p, lattice_type, initial_coords=None):
        """
        lattice_type: str
            can be either "square" or "hexagonal"
        """
        random.seed(random.randint(1, 100000000))

        if(type(size) != int):
            raise TypeError(f'size type should be int but it is {type(size)}')

        self.size = size
        self.p = p
        self.lattice_type = lattice_type
        
        if(lattice_type == 'square'):
            self.knn = 4
            if(initial_coords == None):
                self.coords = set(itertools.product(range(1, self.size+1), repeat=2))
            else:
                self.coords = initial_coords.copy()
            self.find_all_nearest_neighbors()

        elif(lattice_type == 'hexagonal'):
            self.knn = 3

            if(initial_coords == None):
                self.coords = set()
                self.coords.update( itertools.product(range(1, self.size+1, 4), range(1, self.size+1, 2)) )
                self.coords.update( itertools.product(range(2, self.size+1, 4), range(1, self.size+1, 2)) )
                self.coords.update( itertools.product(range(3, self.size+1, 4), range(2, self.size+1, 2)) )
                self.coords.update( itertools.product(range(4, self.size+1, 4), range(2, self.size+1, 2)) )
            else:
                self.coords = initial_coords.copy()
            self.find_all_nearest_neighbors()
        else:
            raise ValueError('Choose correct lattice type)')

        self.occupied = set()
        self.prohibited = set()
        self.boundaries = set()
        self.set_boundaries()
        self.current_nearest_neighbors = set()

        if(lattice_type == 'square'):
            middle_coord = (size // 2, size // 2)
            self.occupied.add(middle_coord)
            self.coords.remove(middle_coord)
            self.current_nearest_neighbors.update(self.all_neighbors[(middle_coord)])
        elif(lattice_type == 'hexagonal'):
            x = size // 2
            y = size // 2
            while((x, y) not in self.coords):
                print(f'while (x, y) = {(x, y)}')
                x += 1
            self.occupied.add((x, y))
            self.coords.remove((x, y))
            self.current_nearest_neighbors.update(self.all_neighbors[(x, y)])



    def find_all_nearest_neighbors(self):
        self.all_neighbors = dict()
        for coord in self.coords:
            neighbors = []
            for neighbor in self.get_neighbors(coord):
                if(neighbor in self.coords):
                    neighbors.append(neighbor)
            self.all_neighbors[coord] = tuple(neighbors)

    def set_boundaries(self):
        for b in itertools.product([1], range(1, self.size)):
            self.boundaries.add(b)
        for b in itertools.product([self.size], range(1, self.size)):
            self.boundaries.add(b)
        for b in itertools.product(range(1, self.size + 1), [1]):
            self.boundaries.add(b)
        for b in itertools.product(range(1, self.size + 1), [self.size]):
            self.boundaries.add(b)
    
    def get_neighbors(self, coord):
        """
        coord: tuple
            (x, y)
        """
        if(self.lattice_type == 'square'):
            return ((coord[0] + 1, coord[1]), (coord[0], coord[1] + 1), (coord[0] - 1, coord[1]), (coord[0], coord[1] - 1))
        elif(self.lattice_type == 'hexagonal'):
            if(coord[0] % 2 == 0):
                return ((coord[0] - 1, coord[1]), (coord[0]+1, coord[1]-1), (coord[0]+1, coord[1]+1))
            else:
                return ((coord[0] + 1, coord[1]), (coord[0]-1, coord[1]-1), (coord[0]-1, coord[1]+1))

    def occupy_nodes(self, coords):
        """
        coords: tuple
            coordinates (x, y)
        """
        self.occupied.update(coords)
        self.coords.difference_update(coords)

    def prohibit_nodes(self, coords):
        self.prohibited.update(coords)
        self.coords.difference_update(coords)

    def make_step(self):
        if(len(self.current_nearest_neighbors) == 0):
            return 0

        random_neighbors = random.sample(self.current_nearest_neighbors,
                                         k=round(self.p * len(self.current_nearest_neighbors)))

        self.prohibit_nodes(self.current_nearest_neighbors.difference(random_neighbors))
        self.occupy_nodes(random_neighbors)

        self.current_nearest_neighbors.clear()
        allowed_neighbors = []
        for rn in random_neighbors:
            for neighbor in self.all_neighbors[rn]:
                if(neighbor not in self.prohibited and neighbor not in self.occupied):
                    allowed_neighbors.append(neighbor)
        self.current_nearest_neighbors.update(allowed_neighbors)
        return 1

    def run(self):

        switch = 1
        while(len(self.coords) > 0 and switch == 1):
            switch = self.make_step()

    def is_perc(self):
        return bool( len(self.occupied.intersection(self.boundaries)) )

    def get_fractal_dim(self, plot=False):
        x = []
        y = []
        for coord in self.occupied:
            x.append(coord[0])
            y.append(coord[1])
        x_center = sum(x) / len(x)
        y_center = sum(y) / len(y)

        rx = np.array(x) - x_center
        ry = np.array(y) - y_center
        r = np.sqrt( rx**2 + ry**2 )
        r.sort()
        r_step = r[1::40]

        a = []
        for r_st in r_step:
            count = (r < r_st).sum()
            a.append((r_st, count))
        a = np.array(a)
        log_r = np.log(a[1:, 0])[:-1000]
        log_c = np.log(a[1:, 1])[:-1000]

        if(plot):
            plt.scatter(log_r, log_c, s=0.8)
            plt.grid(True)
            plt.show()

        return np.polyfit(log_r, log_c, 1)[0]

    def plot(self, draw_prohibited=False):
        x = []
        y = []
        system = np.zeros(shape=(self.size, self.size))
        for coord in self.occupied:
            x.append(coord[0]-1)
            y.append(coord[1]-1)
            system[coord[0]-1, coord[1]-1] = 1


        x_occ = []
        y_occ = []
        prohibited = np.zeros(shape=(self.size, self.size))
        for coord in self.prohibited:
            x_occ.append(coord[0]-1)
            y_occ.append(coord[1]-1)
            prohibited[coord[0]-1, coord[1]-1] = -1

        cdict = [
            (255/255, 34/255, 0/255),
            (34/255, 139/255, 34/255)]

        newcmap = LinearSegmentedColormap.from_list('myCmap', cdict, N=2)
        
        plt.subplots(figsize=(8, 6))
        if(draw_prohibited):
            plt.scatter(x=np.array(x_occ), y=np.array(y_occ), c=prohibited[x_occ, y_occ], s=5, cmap=newcmap)

        plt.scatter(x=np.array(x), y=np.array(y), c=system[x, y], s=0.5, cmap=newcmap)
        # plt.imshow(system)
        plt.clim(-1, 1)
        plt.colorbar()
        plt.show()


if(__name__ == '__main__'):

    t1 = datetime.datetime.now()

    cluster = PercCluster(0.6, 'square')
    # cluster = PercCluster(0.75, 'hexagonal')
    cluster.run()
    print(f'total time = {datetime.datetime.now() - t1}')
    cluster.plot(draw_prohibited=True)



    
        



