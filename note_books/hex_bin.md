```
import numpy as np
```


```
MAP = {
"max_lat": 47.7776,
"min_lat": 47.1559,
"span_lat": 0.6217,
"max_long": -121.315,
"min_long": -122.519,
"span_long": 1.204
}
```


```
DM_LAT = MAP['span_lat']*np.random.random(20000)
DM_LONG = MAP['span_long']*np.random.random(20000)
BASE_H = 350
BASE_W = 540
```


```
def hex_bin(LAT, LONG, base_h, base_w):
    span = (LAT.max() - LAT.min(), LONG.max() - LONG.min())
    base_w = round((span[1] - span[0]) * base_h)
    for (lat, lon) in zip(LAT, LONG):
        lat_bins =  int(round((LAT.max() - lat) * base_h))
        long_bins = int(round((LONG.max() - lon) * base_w))
        yield (lat_bins, long_bins)
```

def main():
    count = 0 
    for num in hex_bin(DM_LAT, DM_LONG, 350, 640):
        if count > 49:
            break
        print(num)
        count += 1


```
if __name__ == '__main__':
    main()
```
