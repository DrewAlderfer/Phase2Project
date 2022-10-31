```
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from tabulate import tabulate
```


```
def make_array(num, div, opt=None, verbose=False):
    def v_print(msg_string):
        if verbose is True:
            return print(msg_string)

    rows, rem = divmod(num, div)
    if rem == 0:
        result = []
        for row in range(rows):
            row_out = []
            for col in range(div):
                # print(row_out)
                row_out.append(col)
            result.append(row_out)
        return result

    v_print(f"{div} does not go into {num} and equal amount of times:")
    v_print(f"{num}/{div} returns {rows} with {rem} remaining.")
    div_up, div_down = (div, div)
    while div_up < 1000:
        if opt in ("+", None):
            div_up += 1
        if opt in ("-", None) and div_down > 0:
            div_down -= 1
        v_print(f"trying {num}/{div_up}")
        if divmod(num, div_up)[1] == 0:
            v_print(f"success! recursing with {num} and {div_up}")
            return make_array(num, div_up, verbose=verbose)
        v_print(f"trying {num}/{div_down}")
        if divmod(num, div_down)[1] == 0:
            v_print(f"success! recursing with {num} and {div_down}")
            return make_array(num, div_down, verbose=verbose)
```


```
# This is all code from a jupyter notebook that needs to be refactored into a useable tool
house_num_col = list(house_num.columns)
column_name = iter(house_num_col)
color = iter(colors)
num_of_columns = len(house_num_col)
graph_shape = make_array(num_of_columns, 4)
fig, ax = plt.subplots(len(graph_shape), len(graph_shape[0]), figsize=(12, 12))
ax[0, 0].hist(x=house_num['price'])
fig.set_edgecolor(color='black')
fig.set_tight_layout(tight=True)
for row, obj in enumerate(graph_shape):
    for col in obj:
        name = next(column_name)
        ax[row, col].hist(x=house_num[name], bins=16, color=next(color), alpha=.4)
        ax[row, col].set_title(name)
        # print(row, col, next(column_gen), next(color))
plt.show()
```
