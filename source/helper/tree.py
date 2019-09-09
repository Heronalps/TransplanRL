import numpy as np

class Node:
    def __init__(self, position=-1, priority=0, end=-1):
        self.left = None
        self.right = None
        self.parent = None
        self.position = position
        self.priority = priority
        self.end = end

    def hasChildren(self):
        if (self.right == None and self.left == None):
            return False
        return True

class SumTree:
    def __init__(self, size):
        
        self._root = Node()
        size_left = int(size/2)
        # Initialization of the tree
        self._root.left = self._createSubtree(self._root, 0, size_left) # [a,b[
        self._root.right = self._createSubtree(self._root, size_left, size)
        self._max_priority = 1

    def _createSubtree(self, parent, begin, end):
        
        n_elem = end - begin
        if (n_elem == 1):
             node = Node(position=begin)
             node.parent = parent
             node.end = end
             return node

        # At least 2 values (leaves) left
        mid = int((end + begin)/2)
        node = Node(end=end)
        node.parent = parent
        node.left = self._createSubtree(node, begin, mid)
        node.right = self._createSubtree(node, mid, end)
        return node

    def update(self, index, priority=-1):
        
        if (priority == -1):
            priority = self._max_priority
        elif (priority > self._max_priority):
            self._max_priority = priority

        # Search for index
        node = self.findIndex(index)

        # Replace with new priority
        diff = priority - node.priority
        node.priority = priority

        # Update value
        self._updateValue(node.parent, diff)

    def _updateValue(self, node, diff):
        node.priority += diff
        if (node.parent != None):
            self._updateValue(node.parent, diff)

    def findIndex(self, index):
        
        if(self._root != None):
            return self._findIndex(index, self._root)
        else:
            return None

    def _findIndex(self, index, node):
        if (node.position == index):
            return node

        if (index < node.left.end):
            return self._findIndex(index, node.left)
        else:
            return self._findIndex(index, node.right)

    def getBatch(self, n, rng, dataset):
        
        pmax = self._root.priority
        step = pmax / n
        indices = np.zeros(n, dtype='int32')
        for i in range(n):
            p = rng.uniform(i*step, (i+1)*step)
            node = self.find(p)
            index = self._checkTerminal(node.position, dataset)
            if (index >= 0):
                indices[i] = index
            else:
                return np.zeros(0)

        return indices

    def _checkTerminal(self, index, dataset):
        
        history_size = dataset._max_history_size
        terminals = dataset._terminals
        n_elems = dataset.n_elems

        lower_bound = history_size - 1

        # Check if the index is valid wrt terminals
        first_try = index
        start_wrapped = False
        while True:
            i = index - 1
            processed = 0
            for _ in range(history_size - 1):
                if (i < 0 or terminals[i]):
                    break

                i -= 1
                processed += 1

            if (processed < history_size - 1):
                # if we stopped prematurely, shift slice to the left and try again
                index = i
                if (index < lower_bound):
                    start_wrapped = True
                    index = n_elems - 1
                if (start_wrapped and index <= first_try):
                    return -1
            else:
                # else index was ok according to terminals
                return index

    def find(self, priority):
        
        if(self._root != None):
            return self._find(priority, self._root)
        else:
            return None

    def _find(self, priority, node):
        if (not node.hasChildren()):
            return node

        if(priority <= node.left.priority):
            return self._find(priority, node.left)
        else:
            return self._find(priority - node.left.priority, node.right)

    def printTree(self):
    # Classical printout method. Mostly for debugging purposes.
        if(self._root != None):
            self._printTree(self._root)

        print("===============")

    def _printTree(self, node):
        if(node != None):
            self._printTree(node.left)
            print(node.position, node.priority)
            self._printTree(node.right)
        

if __name__ == "__main__":
    pass
