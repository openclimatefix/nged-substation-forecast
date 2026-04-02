import numpy as np

lons = np.arange(-180.0, 180.0, 0.25)
new_lons = ((lons + 180) % 360) - 180

diff = np.abs(lons - new_lons)
print("Max difference:", diff.max())
print("Are all equal?", np.allclose(lons, new_lons))
