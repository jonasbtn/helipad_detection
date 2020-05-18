from helipad_detection.src.utils.globalmaptiles import GlobalMercator


if __name__ == "__main__":
    globalmercator = GlobalMercator()

    ( minLat, minLon, maxLat, maxLon ) = globalmercator.TileLatLonBounds(265454, 343865, 19)

    print(minLat)
    print(minLon)

    print(maxLat)
    print(maxLon)

    meanLat = (minLat+maxLat)/2
    meanLon = (minLon+maxLon)/2

    print(meanLat)
    print(meanLon)

    coordinates = globalmercator.TileLatLonBox(265454, 343865, 19)

    for c in coordinates:
        print(c)

