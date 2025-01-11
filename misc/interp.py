# {0.94475, 0.00, 0.00066, 0.00}
# {0.00, -1.73205, -0.00067, 0.00}
# {0.00, 0.00, 1.00503, -1.00503}
# {0.00, 0.00, 1.00, 0.00}

import numpy as np

# ndcZ = (1.00503 * clipZ - 1.00503)/clipZ
# clipZ*ndcZ = 1.00503 * clipZ - 1.00503
# clipZ*(ndcZ-1.00503) = -1.00503
# clipZ = -1.00503 / (ndcZ - 1.00503)

matrix = np.array([[0.94475, 0.00, 0.00066, 0.00],
                    [0.00, -1.73205, -0.00067, 0.00],
                    [0.00, 0.00, 1.00503, -1.00503],
                    [0.00, 0.00, 1.00, 0.00]])


def plotNDCX_W_plot(start_pos_vs,end_pos_vs):
    import numpy as np
    start_pos_vs = np.array(start_pos_vs)
    end_pos_vs = np.array(end_pos_vs)

    start_pos_vs = np.dot(matrix, start_pos_vs)
    end_pos_vs = np.dot(matrix, end_pos_vs)

    print(start_pos_vs)
    print(end_pos_vs)

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as patches

    fig = plt.figure()

    start_ndc = np.dot(matrix, start_pos_vs)
    #start_ndc = start_ndc / start_ndc[3]

    end_ndc = np.dot(matrix, end_pos_vs)
    #end_ndc = end_ndc / end_ndc[3]
    print("ST,ED",start_ndc, end_ndc)

    for i in range(100):
        interp_pos = start_pos_vs + (end_pos_vs - start_pos_vs) * i / 100
        interp_invw = 1.0/start_ndc[3] + (1.0/end_ndc[3] - 1.0/start_ndc[3]) * i / 100
        interp_w = start_ndc[3] + (end_ndc[3] - start_ndc[3]) * i / 100
        interp_ndc = start_ndc + (end_ndc - start_ndc) * i / 100
        interp_ndc = interp_ndc / interp_ndc[3]

        ndc_pos = np.dot(matrix, interp_pos)
        ndc_pos = ndc_pos / ndc_pos[3]

        plt.scatter(ndc_pos[0], ndc_pos[2]-interp_ndc[2],c='b')    
    plt.show()

def test(start_vs,ori_vs):
    clip_start = np.dot(matrix, start_vs)
    clip_ori = np.dot(matrix, ori_vs)

    clip_proc = np.dot(matrix, clip_start+clip_ori)

    ndc_start_w = clip_start[3]
    ndc_proc_w = clip_proc[3]

    ndc_start = clip_start / clip_start[3]
    ndc_proc = clip_proc / clip_proc[3]

    import matplotlib.pyplot as plt

    ndc_pos_lst = []
    clip_pos_lst = []
    clip_pos_lst_2 = []
    plt.figure()
    for i in range(200):
        ndc_pos = ndc_start + (ndc_proc - ndc_start) * i / 20
        ndc_pos_w = ndc_start_w + (ndc_proc_w - ndc_start_w) * i / 20

        # 
        #ndc_pos[2]=1
        clip_pos = np.dot(np.linalg.inv(matrix), ndc_pos*ndc_pos_w)
        clip_pos = clip_pos / clip_pos[3]

        print(clip_pos)

        ndc_pos_lst.append(ndc_pos)
        clip_pos_lst.append(clip_pos)

    
    for i in range(199):
        cur_ps = clip_pos_lst[i]
        next_ps = clip_pos_lst[i+1]

        slope = (next_ps[2] - cur_ps[2]) / np.sqrt((next_ps[0] - cur_ps[0])**2 + (next_ps[1] - cur_ps[1])**2)
        cur_z = cur_ps[2]

        to_ndcz = (1.00503 * cur_z - 1.00503)/cur_z
        plt.scatter(i,to_ndcz,c='b')  
        #plt.scatter(i,cur_ps[2],c='c')  
    plt.show()


# for viewspace pts V1 and V2, projection matrix P, and clip space pts C1 and C2
# C1.z = (a * V1.z - b)/V1.z = a - b/V1.z
# C2.z = (a * V2.z - b)/V2.z = a - b/V2.z
# C1.x = k * V1.x/V1.z + jitterX / V1.z
# C2.x = k * V2.x/V2.z + jitterX / V2.z


def test2():
    test_pt = np.array([1,0.5,3,1])

    clip_pt = np.dot(matrix, test_pt)
    clip_pt = clip_pt / clip_pt[3]

    inv_matrix = np.linalg.inv(matrix)

    for i in range(50):
        mul_w = i/50
        rev_pt = np.dot(inv_matrix, clip_pt * mul_w)
        rev_pt = rev_pt / rev_pt[3]
        print(rev_pt)

if __name__ == '__main__':
    test2()