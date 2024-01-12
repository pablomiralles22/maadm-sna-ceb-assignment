class DisjointSetUnion:
    def __init__(self, n: int):
        self.parent = [i for i in range(n)]
    
    def find(self, x: int) -> int:
        if self.parent[x] == x:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def join(self, x: int, y: int) -> None:
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        self.parent[y] = x
    
    def get_components(self) -> list[list[int]]:
        components = {}
        for idx in range(len(self.parent)):
            components.setdefault(self.find(idx), []).append(idx)
        return list(components.values())
