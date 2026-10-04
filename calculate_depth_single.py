# Implemented by Zhuoyu Yang, Matsushita Lab., Osaka University, Mar. 30th 2020.
# Modified at July 20th 2020. Support adding depth prior.
import cv2
import argparse
import numpy as np
import scipy.sparse as sparse
from time import time
from scipy.signal import convolve2d
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import spsolve
# import sparseqr
import cupy as cp
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as clinalg
HAS_CUPY = True

# command line parser
parser = argparse.ArgumentParser(description='Normal Integration by mesh deformation')
parser.add_argument('normal', help='the path of normal map')
parser.add_argument('-d', '--depth', default=None, help='the path of depth prior')
parser.add_argument('--d_lambda', type=float, default=0.1, help='how much will the depth prior influence the result')
parser.add_argument('-o', '--output', default='output', help='name of the output object and depth map')
parser.add_argument('--vertex_depth', dest='depth_type', action='store_const',
                    const='vertex', default='pixel', help='output vertex depth map, by default pixel depth map')
parser.add_argument('--obj', dest='write_obj', action='store_const',
                    const=True, default=False, help='write wavefront obj file, by default False')
parser.add_argument('--spsolve', dest='spsolve', action='store_const',
                    const=True, default=True,help="""For those who don't have sparseqr, use spsolve from scipy. 
                    Due to memory issues, please ensure your normal map is smaller than 2k*2k. 
                    And spsolve is much more slower than sparseqr""")
parser.add_argument('--refine_depth', dest='refine_depth', action='store_const',
                    const=True, default=False,help="""Refine depth prior by normal map. Do normal integration 
                    with normal map only. Use the result depth map to remove outlier depth priors.""")

def write_depth_map(filename, depth, mask, v_mask, depth_type='pixel'):
    if depth_type == 'pixel':
        cov_mask = np.array([0.25,0.25,0.25,0.25]).reshape(2,2)
        depth = convolve2d(depth, cov_mask, mode="valid") # keep the resolution of depth map same with the input normal map
        depth[~mask] = np.nan
    else:
        depth[~v_mask] = np.nan
    np.save(filename, depth)


def write_obj(filename, d, d_ind):
    obj = open(filename, "w")
    h, w = d.shape

    x = np.arange(0.5, w, 1)
    y = np.arange(h-0.5, 0, -1)
    xx, yy = np.meshgrid(x, y)
    mask = d_ind>0
    
    xyz = np.vstack((xx[mask], yy[mask], d[mask])).T
    obj.write(''.join(["v {0} {1} {2}\n".format(x, y, z) for x, y, z in xyz])) # write vertices into obj file

    right = np.roll(d_ind, -1, axis=1)
    right[:, -1] = 0
    right_mask = right>0

    down = np.roll(d_ind, -1, axis=0)
    down[-1, :] = 0
    down_mask = down>0

    rd = np.roll(d_ind, -1, axis=1)
    rd = np.roll(rd, -1, axis=0)
    rd[-1, :] = 0
    rd[:, -1] = 0
    rd_mask = rd>0

    up_tri = mask&rd_mask&right_mask # counter clockwise
    low_tri = mask&down_mask&rd_mask # counter clockwise

    xyz = np.vstack((d_ind[up_tri], rd[up_tri], right[up_tri])).T
    obj.write(''.join(["f {0} {1} {2}\n".format(x, y, z) for x, y, z in xyz])) # write upper triangle facet into obj file
    xyz = np.vstack((d_ind[low_tri], down[low_tri], rd[low_tri])).T
    obj.write(''.join(["f {0} {1} {2}\n".format(x, y, z) for x, y, z in xyz])) # write lower triangle facet into obj file

    obj.close()


def substract_offset(d):
    return d-np.nanmean(d)

class Normal_Integration(object):

    def __init__(self, nomral_path, depth_path=None, d_lambda=0.1, solver='sparseqr', refine_depth=False): # you can tune d_lambda to decide how much the depth prior will influence the result.
        self.solver = solver
        self.n, self.mask = self.read_normal_map(nomral_path) # read normal map and background mask
        # sel.n[]
        self.N = None # mean-substraction matrix N defined in equation (5), "Surface-from-Gradients: An Approach Based on Discrete Geometry Processing"
        self.d_lambda = d_lambda # lambda for depth prior

        self.ilim, self.jlim = self.mask.shape # size of normal map and mask
        self.mesh_count = (self.mask.astype(np.int32)).sum() # how many pixels in the normal map

        self.lu = np.pad(self.mask, [(0, 1), (0, 1)], mode='constant') # mask for left upper vertices
        self.ru = np.pad(self.mask, [(0, 1), (1, 0)], mode='constant') # mask for right upper vertices
        self.ld = np.pad(self.mask, [(1, 0), (0, 1)], mode='constant') # mask for left down vertices
        self.rd = np.pad(self.mask, [(1, 0), (1, 0)], mode='constant') # mask for right down vertices

        self.v_mask = self.lu|self.ru|self.ld|self.rd # mask for all vertices
        self.v_count = (self.v_mask.astype(np.int32)).sum() # total number of all vertices

        self.v_index = np.zeros_like(self.v_mask, dtype='uint') # indices for all vertices
        self.v_index[self.v_mask] = np.arange(self.v_count)+1

        self.v_depth = np.zeros_like(self.v_index, dtype=np.float32) # depth for the vertices

        self.A, self.b = self.construct_Ax_b_without_depth_prior()

        # below we are going to solve NAx=Nb, let's construct them first
        self.N = self.construct_N()
        self.NA = self.N@self.A
        self.Nb = self.N@self.b

        if self.solver == 'spsolve' or 'cupy':
            self.NATNA = (self.NA.T@self.NA).tocsc()
            self.NATNb = self.NA.T@self.Nb


        
        if depth_path is not None: # add depth prior
            d_prior = np.load(depth_path)
            d_prior[self.mask==False]=np.nan
            if refine_depth:
                self.mesh_deformation()
                cov_mask = np.array([0.25,0.25,0.25,0.25]).reshape(2,2)
                d_from_normal_integration = convolve2d(self.v_depth, cov_mask, mode="valid") # keep the resolution of depth map same with the input normal map
                d_from_normal_integration[~self.mask] = np.nan

                err = np.abs(substract_offset(d_prior)-substract_offset(d_from_normal_integration))
                err_mask = err > 10*np.nanmean(err)
                d_prior[err_mask] = np.nan

            A, b = self.add_depth_prior(d_prior)
            self.NA = sparse.vstack([self.NA, A])
            self.Nb = np.vstack([self.Nb, b])

        if self.solver == 'spsolve' or 'cupy':
            self.NATNA = (self.NA.T@self.NA).tocsc()
            self.NATNb = self.NA.T@self.Nb




    def add_depth_prior(self, d):
         # read depth prior
        d_mask = ~np.isnan(d) # mask for depth prior
        d_count = (d_mask.astype(np.int32)).sum() # total number of depth prior

        # scipy.sparse.coo_matrix((data, (row, col))
        # build row for sparse coo matrix
        row = np.arange(d_count).repeat(4)

        # build col for sparse coo matrix
        lu = np.pad(d_mask, [(0, 1), (0, 1)], mode='constant') # mask for left upper vertices
        ru = np.pad(d_mask, [(0, 1), (1, 0)], mode='constant') # mask for right upper vertices
        ld = np.pad(d_mask, [(1, 0), (0, 1)], mode='constant') # mask for left down vertices
        rd = np.pad(d_mask, [(1, 0), (1, 0)], mode='constant') # mask for right down vertices

        col = np.vstack((self.v_index[lu],self.v_index[ru],self.v_index[ld],self.v_index[rd])).T.ravel()-1 #should -1 to start from 0

        # build data for sparse coo matrix
        data = np.ones(4*d_count)
        data = 1/4*data*self.d_lambda
        
        # build A for Ax=b
        A = sparse.coo_matrix((data, (row, col)), shape=(d_count, self.v_count))
        # build b for Ax=b
        b = d[d_mask].reshape(-1,1)*self.d_lambda
        
        return A, b

    def construct_Ax_b_without_depth_prior(self):

        # scipy.sparse.coo_matrix((data, (row, col))
        # build row for sparse coo matrix
        row = np.arange(4*self.mesh_count)
        # build col for sparse coo matrix
        col = np.vstack((self.v_index[self.lu],self.v_index[self.ru],self.v_index[self.ld],self.v_index[self.rd])).T.ravel()
        # build data for sparse coo matrix
        data = np.ones(4*self.mesh_count)

        x = self.n[...,0][self.mask]
        y = self.n[...,1][self.mask]
        z = -self.n[...,2][self.mask]
        # build A for Ax=b
        A = sparse.coo_matrix((data, (row, col-1)), shape=(4*self.mesh_count, self.v_count))
        # build b for Ax=b
        b = np.vstack(((x/-2+y/2)/z,(x/2+y/2)/z,(x/-2+y/-2)/z,(x/2+y/-2)/z)).T.reshape(-1, 1)

        return A, b

    def read_normal_map(self, path):
        '''
        description:
            read a normal map(jpg, png, bmp, etc.), and convert it to an normalized (x,y,z) form

        input:
            path: path of the normal map

        output:
            n: normalized normal map
            mask: normal map mask
        '''

        if ".npy" in path:
            n = np.load(path)
            print(path)
            mask_bg = (n[...,2] == 0) # get background mask
            normal_map = (n + 1.0) / 2
            # cv2.imshow("Normal Map input", (normal_map[..., [2, 1, 0]] * 255).astype(np.uint8))
        else:
            n = cv2.imread(path)

            n[...,0], n[...,2] = n[...,2], n[...,0].copy() # Change BGR to RGB
            mask_bg = (n[...,2] == 0) # get background mask
            n = n.astype(np.float32) # uint8 -> float32

            # x,y:[0,255]->[-1,1] z:[128,255]->[0,1]
            n[...,0] = n[...,0]*2/255-1
            n[...,1] = n[...,1]*2/255-1
            n[...,2] = (n[...,2]-128)/127
            # n = n*2/255-1
            n = normalize(n.reshape(-1,3)).reshape(n.shape)

        # n[...,0] = n[...,0]*pixmm # convert nx to mm/pixel
        # n[...,1] = n[...,1]*pixmm # convert ny to mm/pixel

        n[...,0] = n[...,0]
        n[...,1] = n[...,1]

        # background_pixel_count = np.count_nonzero(mask_bg)

        # fill background with [0,0,0]
        # n[mask_bg] = [0,0,0]
        return n, ~mask_bg

    def construct_N(self):
        N_ = np.eye(4)-0.25
        data = np.array([N_ for i in range(self.mesh_count)])
        indptr = range(self.mesh_count+1)
        indices = range(self.mesh_count)
        N = sparse.bsr_matrix((data,indices,indptr))
        return N

    def mesh_deformation(self):
        """
        执行网格变形，即求解线性方程组来计算顶点深度。
        """
        if HAS_CUPY and self.solver == 'cupy':
            print("Using GPU (CuPy) Iterative Solver (CG)...")
            try:
                # 1. 显式将 CPU 的 scipy 矩阵转换为 GPU 的 cupy CSR 矩阵
                # self.NATNA 在 CPU 上通常是 CSC 格式，转为 CSR 更适合矩阵乘法
                start_trans = time()
                A_gpu = csp.csr_matrix(self.NATNA)
                b_gpu = cp.array(self.NATNb.flatten())  # 确保 b 是 1D 数组
                # print(f"Data transfer time: {time() - start_trans:.4f}s")

                # 2. 使用共轭梯度法 (Conjugate Gradient) 求解
                # spsolve (直接法) 在 GPU 上极慢且显存消耗大
                # cg (迭代法) 非常适合这种稀疏对称正定系统
                start_solve = time()
                x_gpu, info = clinalg.cg(A_gpu, b_gpu, tol=1e-6, maxiter=5000)
                # print(f"GPU Solve time: {time() - start_solve:.4f}s")

                if info != 0:
                    print(f"Warning: CG did not converge (info={info})")

                # 3. 转回 CPU
                x = cp.asnumpy(x_gpu)
                self.v_depth[self.v_mask] = x.reshape(-1)

            except Exception as e:
                print(f"CuPy solver failed: {e}. Falling back to scipy spsolve.")
                import traceback
                traceback.print_exc()
                # 回退到 CPU
                x = spsolve(self.NATNA, self.NATNb)
                self.v_depth[self.v_mask] = x.reshape(-1)

        elif self.solver == 'sparseqr':
            x = sparseqr.solve(self.NA, self.Nb) # solve NAx = Nb by PySPQR
        else:
            x = spsolve(self.NATNA, self.NATNb)  # solve NATNAx = NATNb by Scipy spsolve
        self.v_depth[self.v_mask] = x.reshape(-1)

if __name__ == '__main__':

    # pixmm = 0.014259    # 自研食指
    # pixmm = 0.019092 # 拾玥食指1号
    pixmm = 0.029769  # 三指1号

    args = parser.parse_args()

    start = time()

    print("Start reading input data...")
    if args.spsolve:
        from scipy.sparse.linalg import spsolve # for those who don't have sparseqr
        task = Normal_Integration(args.normal, pixmm, args.depth, args.d_lambda, 'spsolve', refine_depth=args.refine_depth)
    else:
        import sparseqr
        task = Normal_Integration(args.normal, pixmm, args.depth, args.d_lambda, refine_depth=args.refine_depth) # by default use sparseqr which has a better performance
    print("Start normal integration...")
    task.mesh_deformation()

    end = time()
    print("Time elapsed: {:0.2f}".format(end - start))

    if args.write_obj:
        print("Start writing obj file...")
        write_obj("{0}.obj".format(args.output), task.v_depth, task.v_index) # write obj file

    print("Start writing depth map...")
    write_depth_map("{0}_depth.npy".format(args.output), task.v_depth, task.mask, task.v_mask, depth_type=args.depth_type) # write depth file
    depth=task.v_depth
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
    depth_vis = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imshow("depth_vis", depth_vis)


    def depth_to_normals(depth):
        h, w = depth.shape
        normals = np.zeros((h, w, 3), dtype=np.float32)

        # 对每个像素计算法线
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                # 获取深度图的邻域
                dzdx = (depth[y, x + 1] - depth[y, x - 1]) / 2.0
                dzdy = (depth[y + 1, x] - depth[y - 1, x]) / 2.0

                normal = np.array([-dzdx, dzdy, 1.0])
                normal = normal / np.linalg.norm(normal)  # 单位化法线

                normals[y, x] = normal

        return normals


    normal_map = depth_to_normals(depth, 0.014259)
    normal_map = (normal_map + 1.0) / 2
    # cv2.imshow("Normal Map reverse", (normal_map[..., [2, 1, 0]] * 255).astype(np.uint8))
    cv2.waitKey(0)



    import open3d as o3d
    # 计算点云
    def depth_to_point_cloud(depth, fx, fy, cx, cy):
        h, w = depth.shape
        points = []

        # 将每个像素的深度值转换为 3D 点
        for y in range(h):
            for x in range(w):
                Z = depth[y, x]
                if Z > 0:  # 只考虑深度大于0的有效点
                    X = (x - cx) * Z / fx
                    Y = (y - cy) * Z / fy
                    points.append([X, Y, Z])

        return np.array(points)
    # 创建一个 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    fx = 976.38
    fy = 906.62
    cx = 610.44
    cy = 374.17
    points = depth_to_point_cloud(depth[1:-1,1:-1], fx, fy, cx, cy)
    # 将 numpy 数组转换为 Open3D 点云格式
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points)/2)

    # 假设 pcd 已经准备好
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="完整点云", width=800, height=600)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    # 调整近／远裁剪平面，避免误切
    ctr.set_constant_z_near(0.01)
    ctr.set_constant_z_far(100.0)

    vis.run()
    vis.destroy_window()

    # # 点云法线估计（有助于网格化过程）
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=10))
    #
    # # 使用泊松重建进行网格化
    # print("执行泊松重建...")
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    #
    # # 可选：根据密度值设置网格的可视化效果
    # mesh.compute_vertex_normals()  # 计算网格的法线
    # mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 设置网格颜色
    #
    # # 显示网格化结果
    # print("显示网格化结果...")
    # o3d.visualization.draw_geometries([mesh])
    #
    # print("Finish!")
