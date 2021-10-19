# -*- encoding: utf-8 -*-
import open3d as o3d


def main():
    pcd = o3d.io.read_point_cloud("./ply/C2NCNOH8_gauss.ply")
    print(pcd)
    # o3d.visualization.draw_geometries([pcd])
    downpcd = pcd.voxel_down_sample(voxel_size=0.1)
    print(downpcd)
    # o3d.visualization.draw_geometries([pcd])
    octree = o3d.geometry.Octree(max_depth=6)
    octree.convert_from_point_cloud(pcd)
    o3d.visualization.draw_geometries([octree])


if __name__ == '__main__':
    main()
