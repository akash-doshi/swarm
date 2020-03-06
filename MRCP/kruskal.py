class Kruskal(object):
    """ generated source for class Kruskal """
    nodes = HashSet()

    #  Array of connected components
    allEdges = TreeSet()

    #  Priority queue of Edge objects
    allNewEdges = Vector()

    #  Edges in Minimal-Spanning Tree
    MAX_NODES = int()

    def __init__(self, maxN):
        """ generated source for method __init__ """
        #  Constructor
        self.MAX_NODES = maxN
        self.nodes = [None]*MAX_NODES
        #  Create array for components
        self.allEdges = TreeSet(Edge())
        #  Create empty priority queue
        self.allNewEdges = Vector(MAX_NODES)
        #  Create vector for MST edges

    def initializeGraph(self, A, connect4):
        """ generated source for method initializeGraph """
        rows = A.length
        cols = A[0].length
        #
        #  * Created by atkap on 5/20/2016.
        #
        #  Array of connected components
        #  Priority queue of Edge objects
        #  Edges in Minimal-Spanning Tree
        #  Constructor
        #  Create array for components
        #  Create empty priority queue
        #  Create vector for MST edges
        i = 0
        while i < rows:
            #
            #  * Created by atkap on 5/20/2016.
            #
            #  Array of connected components
            #  Priority queue of Edge objects
            #  Edges in Minimal-Spanning Tree
            #  Constructor
            #  Create array for components
            #  Create empty priority queue
            #  Create vector for MST edges
            #
            #  * Created by atkap on 5/20/2016.
            #
            #  Array of connected components
            #  Priority queue of Edge objects
            #  Edges in Minimal-Spanning Tree
            #  Constructor
            #  Create array for components
            #  Create empty priority queue
            #  Create vector for MST edges
            while j < cols:
                #
                #  * Created by atkap on 5/20/2016.
                #
                #  Array of connected components
                #  Priority queue of Edge objects
                #  Edges in Minimal-Spanning Tree
                #  Constructor
                #  Create array for components
                #  Create empty priority queue
                #  Create vector for MST edges
                if A[i][j]:
                    if i > 0 and A[i - 1][j]:
                        AddToAllEdges(i * cols + j, (i - 1) * cols + j, 1)
                    if i < rows - 1 and A[i + 1][j]:
                        AddToAllEdges(i * cols + j, (i + 1) * cols + j, 1)
                    if j > 0 and A[i][j - 1]:
                        AddToAllEdges(i * cols + j, i * cols + j - 1, 1)
                    if j < cols - 1 and A[i][j + 1]:
                        AddToAllEdges(i * cols + j, i * cols + j + 1, 1)
                    if not connect4:
                        if i > 0 and j > 0 and A[i - 1][j - 1]:
                            AddToAllEdges(i * cols + j, (i - 1) * cols + j - 1, 1)
                        if i < rows - 1 and j < cols - 1 and A[i + 1][j + 1]:
                            AddToAllEdges(i * cols + j, (i + 1) * cols + j + 1, 1)
                        if i > rows - 1 and j > 0 and A[i + 1][j - 1]:
                            AddToAllEdges(i * cols + j, (i + 1) * cols + j - 1, 1)
                        if i > 0 and j < cols - 1 and A[i - 1][j + 1]:
                            AddToAllEdges(i * cols + j, (i - 1) * cols + j + 1, 1)
                j += 1
            i += 1

    def AddToAllEdges(self, from_, to, cost):
        """ generated source for method AddToAllEdges """
        self.allEdges.add(Edge(from_, to, cost))
        if self.nodes[from_] == None:
            self.nodes[from_] = HashSet(2 * MAX_NODES)
            self.nodes[from_].add(int(from_))
        if self.nodes[to] == None:
            self.nodes[to] = HashSet(2 * MAX_NODES)
            self.nodes[to].add(int(to))

    def performKruskal(self):
        """ generated source for method performKruskal """
        size = len(self.allEdges)
        #
        #  * Created by atkap on 5/20/2016.
        #
        #  Array of connected components
        #  Priority queue of Edge objects
        #  Edges in Minimal-Spanning Tree
        #  Constructor
        #  Create array for components
        #  Create empty priority queue
        #  Create vector for MST edges
        #  Update priority queue
        #  Create set of connect components [singleton] for this node
        #  Create set of connect components [singleton] for this node
        i = 0
        while i < size:
            #
            #  * Created by atkap on 5/20/2016.
            #
            #  Array of connected components
            #  Priority queue of Edge objects
            #  Edges in Minimal-Spanning Tree
            #  Constructor
            #  Create array for components
            #  Create empty priority queue
            #  Create vector for MST edges
            #  Update priority queue
            #  Create set of connect components [singleton] for this node
            #  Create set of connect components [singleton] for this node
            if self.allEdges.remove(curEdge):
                #  successful removal from priority queue: allEdges
                if nodesAreInDifferentSets(curEdge.from_, curEdge.to):
                    #  print "Nodes are in different sets ...";
                    if self.nodes[curEdge.from_].size() > nodes[curEdge.to].size():
                        src = nodes[curEdge.to]
                        dst = nodes[dstHashSetIndex = curEdge.from_]
                    else:
                        src = nodes[curEdge.from_]
                        dst = nodes[dstHashSetIndex = curEdge.to]
                    #
                    #  * Created by atkap on 5/20/2016.
                    #
                    #  Array of connected components
                    #  Priority queue of Edge objects
                    #  Edges in Minimal-Spanning Tree
                    #  Constructor
                    #  Create array for components
                    #  Create empty priority queue
                    #  Create vector for MST edges
                    #  Update priority queue
                    #  Create set of connect components [singleton] for this node
                    #  Create set of connect components [singleton] for this node
                    #  successful removal from priority queue: allEdges
                    #  print "Nodes are in different sets ...";
                    #  have to transfer all nodes including curEdge.to
                    #  have to transfer all nodes including curEdge.from
                    while j < transferSize:
                        #
                        #  * Created by atkap on 5/20/2016.
                        #
                        #  Array of connected components
                        #  Priority queue of Edge objects
                        #  Edges in Minimal-Spanning Tree
                        #  Constructor
                        #  Create array for components
                        #  Create empty priority queue
                        #  Create vector for MST edges
                        #  Update priority queue
                        #  Create set of connect components [singleton] for this node
                        #  Create set of connect components [singleton] for this node
                        #  successful removal from priority queue: allEdges
                        #  print "Nodes are in different sets ...";
                        #  have to transfer all nodes including curEdge.to
                        #  have to transfer all nodes including curEdge.from
                        #  move each node from set: src into set: dst
                        #  and update appropriate index in array: nodes
                        if src.remove(srcArray[j]):
                            dst.add(srcArray[j])
                            self.nodes[(int(srcArray[j])).intValue()] = nodes[dstHashSetIndex]
                        else:
                            #  This is a serious problem
                            print "Something wrong: set union"
                            System.exit(1)
                        j += 1
                    self.allNewEdges.add(curEdge)
                else:
            else:
                print "TreeSet should have contained this element!!"
                System.exit(1)
            i += 1

    def nodesAreInDifferentSets(self, a, b):
        """ generated source for method nodesAreInDifferentSets """
        return (not self.nodes[a] == self.nodes[b])

    def printFinalEdges(self):
        """ generated source for method printFinalEdges """
        print "The minimal spanning tree generated by " + "\nKruskal's algorithm is: "
        while not self.allNewEdges.isEmpty():
            print "Nodes: (" + e.from_ + ", " + e.to + ") with cost: " + e.cost
            self.allNewEdges.remove(e)

    def getAllNewEdges(self):
        """ generated source for method getAllNewEdges """
        return self.allNewEdges

    def getAllEdges(self):
        """ generated source for method getAllEdges """
        return self.allEdges

