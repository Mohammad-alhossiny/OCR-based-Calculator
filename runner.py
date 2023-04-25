import draw
import tensors_to_output
from popup import pop_up

d_num = 145
while True:
    draw.init(d_num)
    pop_up(tensors_to_output.init(d_num=d_num))
    d_num += 1
    print(d_num)