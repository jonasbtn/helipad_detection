import math


def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
  return (xtile, ytile)


zoom = 20
lat_deg = 33.53578333333333
lon_deg = -86.81140555555555

(xtile, ytile) = deg2num(lat_deg, lon_deg, zoom)


from utils.globalmaptiles import GlobalMercator

global_mercator = GlobalMercator(tileSize=640)

mx, my = global_mercator.LatLonToMeters(lat_deg, lon_deg)

px, py = global_mercator.MetersToPixels(mx, my, zoom)

print(px)
print(py)

tx, ty = global_mercator.PixelsToTile(px, py)

# gx, gy = global_mercator.GoogleTile(tx,ty, zoom)

print(tx)
print(ty)