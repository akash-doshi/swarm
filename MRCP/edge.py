class Edge(Comparator):
    """ generated source for class Edge """
    #  Inner class for representing edge+end-points
    from_ = int()
    to = int()
    cost = int()

    @overloaded
    def __init__(self):
        """ generated source for method __init__ """
        super(Edge, self).__init__()
        #  Default constructor for TreeSet creation

    @__init__.register(object, int, int, int)
    def __init___0(self, f, t, c):
        """ generated source for method __init___0 """
        super(Edge, self).__init__()
        #  Inner class constructor
        self.from_ = f
        self.to = t
        self.cost = c

    def compare(self, o1, o2):
        """ generated source for method compare """
        #  Used for comparisions during add/remove operations
        cost1 = (Edge(o1)).cost
        cost2 = (Edge(o2)).cost
        from1 = (Edge(o1)).from_
        from2 = (Edge(o2)).from_
        to1 = (Edge(o1)).to
        to2 = (Edge(o2)).to
        if cost1 < cost2:
            return (-1)
        elif cost1 == cost2 and from1 == from2 and to1 == to2:
            return (0)
        elif cost1 == cost2:
            return (-1)
        elif cost1 > cost2:
            return (1)
        else:
            return (0)

    def hashCode(self):
        """ generated source for method hashCode """
        return (Integer.toString(self.cost) + Integer.toString(self.from_) + Integer.toString(self.to)).hashCode()

    def equals(self, obj):
        """ generated source for method equals """
        #  Used for comparisions during add/remove operations
        e = Edge(obj)
        return (self.cost == e.cost and self.from_ == e.from_ and self.to == e.to)
