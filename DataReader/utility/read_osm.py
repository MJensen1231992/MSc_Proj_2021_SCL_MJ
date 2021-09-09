import pyrosm

# Downloading GIS data from aarhus
fp = pyrosm.get_data("aarhus", directory="./../data/osm")

osm = pyrosm.OSM(fp)

drive_net = osm.get_network(network_type="driving")
drive_net.plot()


