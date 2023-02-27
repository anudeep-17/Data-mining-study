# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import statistics

test_1 = [1.1123423512351463265231234,-2.1251345623563241324,3.12335142513634613451345,4.123514513451342,5.23151346136134,4.234123,6.21341324125]
data = np.array(test_1)
print(data.mean())
print(statistics.mean(test_1))