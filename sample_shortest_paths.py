import os
import networkx as nx
import numpy as np
import json
from tqdm import tqdm

connectivity_dir = '../datasets/REVERIE/connectivity'
def distance(pose1, pose2):
    ''' Euclidean distance between two graph poses '''
    return ((pose1['pose'][3]-pose2['pose'][3])**2\
      + (pose1['pose'][7]-pose2['pose'][7])**2\
      + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

scans = [x.strip() for x in open(os.path.join(connectivity_dir, 'scans_hm3d.txt')).readlines()]
print(len(scans))

def process():
    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                        item['pose'][7], item['pose'][11]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G

    shortest_distances = {}
    shortest_paths = {}
    output_data = []
    for scan, G in tqdm(graphs.items()):  # compute all shortest paths
        shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))

    for scan, matrix in tqdm(shortest_paths.items()):
        for start_viewpoint, paths in matrix.items():
            for end_viewpoint, path in paths.items():
                if 5 <= len(path) <= 7:
                    path_id = f"{start_viewpoint}_{end_viewpoint}"
                    tmp_item = {
                        "path_id": path_id,
                        "path": path,
                        "heading": 0.0,
                        "scan": scan
                    }
                    output_data.append(tmp_item)
                    # for tgt in range(36):
                    #     cpitem = copy.deepcopy(tmp_item)
                    #     cpitem['target'] = tgt
                    #     output_data.append(cpitem)
    print(len(output_data))
    with open("your_path/hm3d_800scan_tgt.json", "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=4, ensure_ascii=False)

process()