import numpy as np
class SumTree():
    def __init__(self,capacity:int) -> None:
        self.data_pointer = 0
        self._capacity = capacity
        self._current_capacity = 0
        self.tree = np.zeros(2*capacity-1)
        self.data = np.zeros(capacity,dtype=object)

    
    def add(self,priority:float,data:object):
        tree_index = self.data_pointer + self._capacity-1

        self.data[self.data_pointer] = data
        self.update(tree_index,priority)

        self.data_pointer+=1
        if self.data_pointer>= self._capacity:
            self.data_pointer = 0
        self._current_capacity+=1
        if self._current_capacity > self._capacity:
            self._current_capacity = self._capacity
        
    
    def update(self,tree_index:int,priority:float):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index !=0:
            tree_index = (tree_index-1)//2
            self.tree[tree_index] +=change
        
    def get_leaf(self,v):
        parent_index = 0

        while True:
            left_child_index = 2*parent_index+1
            right_child_index = left_child_index+1

            if left_child_index>= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v<= self.tree[left_child_index]:
                    parent_index = left_child_index
                
                else:
                    v-=self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self._capacity+1
        return leaf_index,self.tree[leaf_index],self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]
    
    @property
    def current_capacity(self):
        return self._current_capacity

class PER():
    def __init__(self, capacity,per_e=0.01,per_a=0.6,per_b=0.4,per_b_increment=0.001) -> None:
        self._per_e =per_e
        self._per_a = per_a
        self._per_b =per_b
        self._per_b_increment = per_b_increment
        self.absolute_error_upper = 1.0

        self.sum_tree = SumTree(capacity=capacity)
    
    def store(self,experience):
        max_priority = np.max(self.sum_tree.tree[-self.sum_tree._capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.sum_tree.add(max_priority,experience)
    
    def sample(self,n):
        memory_b = []

        b_idx , b_ISWeights = np.empty((n,),dtype=np.int32) , np.empty((n,1),dtype=np.float32)
        priority_segment = self.sum_tree.total_priority / n

        self._per_b = np.min([1.0,self._per_b + self._per_b_increment])

        # EDITED
        start = self.sum_tree._capacity - 1 
        end = start + self.sum_tree._current_capacity
        p_min = np.min(self.sum_tree.tree[start:end]) / self.sum_tree.total_priority
        # p_min = np.min(self.sum_tree.tree[-self.sum_tree._capacity:-self.sum_tree._capacity + self.sum_tree._current_capacity]) / self.sum_tree.total_priority
        # p_min = np.min(self.sum_tree.tree[-self.sum_tree._capacity:]) / self.sum_tree.total_priority
        max_weight  = (p_min*n)**(-self._per_b)

        for i in range(n):
            a,b = priority_segment * i ,priority_segment * (i+1)
            value = np.random.uniform(a,b)
            index ,priority , data = self.sum_tree.get_leaf(value)

            sampling_probs = priority / self.sum_tree.total_priority
            b_ISWeights[i,0] = np.power(n*sampling_probs,-self._per_b)/ max_weight

            b_idx[i] = index
            experience = data
            memory_b.append(experience)
        return b_idx , memory_b , b_ISWeights

    def batch_update(self,tree_idx , abs_errors):
        abs_errors+=self._per_e
        clipped_errors = np.minimum(abs_errors,self.absolute_error_upper)
        ps = np.power(clipped_errors,self._per_a)

        for ti , p in zip(tree_idx,ps):
            self.sum_tree.update(ti,p)
    
    @property
    def current_capacity(self)->int:
        return self.sum_tree.current_capacity

