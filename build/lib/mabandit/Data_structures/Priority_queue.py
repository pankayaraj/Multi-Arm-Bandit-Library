#This Priority Queue take maximum value as it's high priority value. If u want the lowest pn use it with a negtive sign

class Priority_queue:

    def __init__(self):
        self.heap_array = []
        self.heap_size = -1


    def get_parent(self, index):
        return self.heap_array[index//2]

    def get_left_child(self, index):
        return self.heap_array[2*index+1]

    def get_right_child(self, index):
        return self.heap_array[2*index+2]

    def get_maximum(self):
        return self.heap_array[0]


    def insert(self, element):
        self.heap_array.append(element)
        self.heap_size += 1
        self.shifup(self.heap_size)

    def shifup(self, index):
        tem_index = index
        while tem_index != 0:
            tem_index = index // 2
            if self.heap_array[index] > self.heap_array[tem_index]:
                temp = self.heap_array[index]
                self.heap_array[index] = self.heap_array[tem_index]
                self.heap_array[tem_index] = temp
                index = tem_index

            else:
                break

    #you can only delete the minimal one
    def delete(self):
        self.heap_array[0] = self.heap_array[-1]
        self.heap_array.pop()
        self.shift_down()

    def shift_down(self):
        right_child = 2
        left_child = 1
        parent = 0
        while True:
            try:
                if self.heap_array[right_child] > self.heap_array[left_child]:
                    min = right_child
                else:
                    min = left_child

                if self.heap_array[parent] >= self.heap_array[min]:
                    break
                else:
                    temp = self.heap_array[parent]
                    self.heap_array[parent] = self.heap_array[min]
                    self.heap_array[min] = temp

                    parent = min
                    right_child = parent*2 + 2
                    left_child = parent*2 + 1
            except:
                break


#test
'''
h = priority_queue()
h.insert(-5)
h.insert(-4)
h.insert(-3)
h.insert(-10)
h.insert(-1)
h.insert(-100)
h.insert(-2)
h.insert(-15)
h.insert(-12)

print(h.heap_array)

h.delete()
print(h.heap_array)
'''