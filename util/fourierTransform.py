import math
import numpy as np

def dft_impl(x:list[complex],proc:int,scale:int)->list[complex]:
    # f(k) = sum(x[i]*exp(-2pi*ik/N))
    # f(k) = sum(x[2p]*exp(-2pi*(2p)k/N) + x[2p+1]*exp(-2pi*(2p+1)k/N))
    # f(k) = sum(x[2i]*exp(-2pi*ik/(N/2)) + exp(-2pi*k/N)*x[2p+1]*exp(-2pi*ik/(N/2)))
    # f(k) = (dft(x0,k)+exp(-2pi*k/N)*dft(x1,k)), k<N/2
    # f(k+N/2) = sum(x[2p]*exp(-2ipi*(2p)(k+N/2)/N) + x[2p+1]*exp(-2ipi*(2p+1)(k+N/2)/N))
    # f(k+N/2) = sum(x[2p]*exp(-2ipi*p/(N/2))*exp(-2ipi*p))
    #          + sum(x[2p+1]*exp(-2ipi*(2p+1)k/N)*exp(-2ipi*p)*exp(-ipi))
    # exp(iv) = cosv+isinv => exp(-ipi)=-1
    # f(k+N/2) =  sum(x[2p]*exp(-2ipi*p/(N/2))) -  sum(x[2p+1]*exp(-2ipi*(2p+1)k/N))
    # f(k+N/2) = dft(x0,k)-exp(-2pi*k/N)*dft(x1,k)
    # ends: N=1 => f(k) = x[i]
    N = len(x)
    if N == 1:
        return [x[0]]
    Ae = x[0::2]
    Ao = x[1::2]
    Fe = dft_impl(Ae,proc,scale)
    Fo = dft_impl(Ao,proc,scale)
    Ra,Rb = [],[]
    for i in range(N//2):
        c = math.cos(proc*2*math.pi*i/N)
        s = math.sin(proc*2*math.pi*i/N)
        a = Fe[i] + Fo[i]*(c+s*1j)
        b = Fe[i] - Fo[i]*(c+s*1j)
        Ra.append(a/scale)
        Rb.append(b/scale)
    Ra.extend(Rb)
    return Ra

def cooley_turkey_iter_dft_impl(x:list[complex],proc:int,scale:float)->list[complex]:
    def bit_reverse(xv:int,n:int):
        ret = 0
        for i in range(n):
            ret = ret * 2 + ((xv>>(i)) & 1)
        return ret
    N = len(x)
    LgN = int(math.log2(N))
    rev_mapping = [bit_reverse(v,LgN) for v in range(N)]
    x0 = [x[rev_mapping[i]] for i in range(N)]
    x1 = [0 for i in range(N)]
    xR = [x0,x1]
    stride = 1
    for k in range(LgN):
        for i in range(N//(stride*2)):
            lpos = i * (stride*2)
            for j in range(stride):
                vL = xR[k%2][lpos+j]
                vR = xR[k%2][lpos+j+stride]
                c = math.cos(proc*2*math.pi*j/(stride*2))
                s = math.sin(proc*2*math.pi*j/(stride*2))
                xR[1-k%2][lpos+j] = (vL + vR*(c+s*1j))/scale
                xR[1-k%2][lpos+j+stride] = (vL - vR*(c+s*1j))/scale
        stride *= 2
    return [xR[LgN%2][i] for i in range(N)]

def stockham_iter_dft_impl(x:list[complex],proc:int,scale:float)->list[complex]:
    N = len(x)
    Hn = N//2
    LgN = int(math.log2(N))
    x0 = [v for v in x]
    x1 = [0 for v in x]
    xR = [x0,x1]
    stride = 1
    for k in range(LgN):
        for i in range(N//(stride*2)):
            lpos = i*stride
            lpos2 = i*(stride*2)
            for j in range(stride):
                vL = xR[k%2][lpos+j]
                vR = xR[k%2][lpos+j+Hn]
                c = math.cos(proc*2*math.pi*j/(stride*2))
                s = math.sin(proc*2*math.pi*j/(stride*2))
                xR[1-k%2][lpos2+j] = (vL + vR*(c+s*1j))/scale
                xR[1-k%2][lpos2+j+stride] = (vL - vR*(c+s*1j))/scale
        stride *= 2
    return [xR[LgN%2][i] for i in range(N)]


def idft(x:list[complex])->list[float]:
    # x[k] = sum(f[k]*exp(2ipi*k/N))
    return dft_impl(x,1,2)

def dft(x:list[float])->list[complex]:
    return dft_impl(x,-1,1)

def ctidft(x:list[complex])->list[float]:
    return cooley_turkey_iter_dft_impl(x,1,2)

def ctdft(x:list[float])->list[complex]:
    return cooley_turkey_iter_dft_impl(x,-1,1)

def sidft(x:list[complex])->list[float]:
    return stockham_iter_dft_impl(x,1,2)

def sdft(x:list[float])->list[complex]:
    return stockham_iter_dft_impl(x,-1,1)

def fftshift(x:np.ndarray)->np.ndarray:
    ret = np.zeros_like(x,dtype=np.complex128)
    rows, cols = x.shape
    for r in range(rows):
        for c in range(cols):
            ret[r,c] = x[(r+rows//2)%rows,(c+cols//2)%cols]
    return ret

def ifftshift(x:np.ndarray)->np.ndarray:
    ret = np.zeros_like(x,dtype=np.complex128)
    rows, cols = x.shape
    for r in range(rows):
        for c in range(cols):
            ret[r,c] = x[(r-rows//2)%rows,(c-cols//2)%cols]
    return ret

def sdft2(x:np.ndarray)->np.ndarray:
    ret = np.zeros_like(x,dtype=np.complex128)
    intm = np.zeros_like(x,dtype=np.complex128)
    rows, cols = x.shape
    for r in range(rows):
        intm[r,:] = np.array(sdft(x[r]))
    for c in range(cols):
        ret[:,c] = np.array(sdft(intm[:,c]))
    return ret

def sidft2(x:np.ndarray)->np.ndarray:
    ret = np.zeros_like(x)
    intm = np.zeros_like(x)
    rows, cols = x.shape
    for c in range(cols):
        intm[:,c] = np.array(sidft(x[:,c]))
    for r in range(rows):
        ret[r,:] = np.array(sidft(intm[r]))
    return ret

def fftconv(x:np.ndarray,y:np.ndarray)->np.ndarray:
    yh, yw = y.shape
    xh, xw = x.shape
    assert yh == yw and xh == xw, "Only support square kernel"
    image_padding = yh//2   
    image_pad = np.pad(x,((image_padding,image_padding),(image_padding,image_padding)),"constant")
    
    pad_0_low = image_pad.shape[0] // 2 - yh // 2
    pad_0_high = image_pad.shape[0] - yh - pad_0_low
    pad_1_low = image_pad.shape[1] // 2 - yw // 2
    pad_1_high = image_pad.shape[1] - yw - pad_1_low
    kernel_pad = np.pad(y,((pad_0_low,pad_0_high),(pad_1_low,pad_1_high)),"constant")

    # pad to 512
    image_pad2 = np.zeros((512,512))
    kernel_pad2 = np.zeros((512,512))

    # put image_pad to image_pad2, in center
    image_pad2[256-128:256+128,256-128:256+128] = image_pad
    kernel_pad2[256-128:256+128,256-128:256+128] = kernel_pad


    kernel_shift = ifftshift(kernel_pad2)
    
    print(image_pad.shape,kernel_shift.shape)
    img_fft = sdft2(image_pad2)
    kernel_fft = sdft2(kernel_shift)
    conv = img_fft * kernel_fft
    print(kernel_fft)  
    conv = sidft2(conv)
    return conv[256-128:256+128,256-128:256+128]

def fftconvnp(x:np.ndarray,y:np.ndarray)->np.ndarray:
    yh, yw = y.shape
    xh, xw = x.shape
    assert yh == yw and xh == xw, "Only support square kernel"
    image_padding = yh//2   
    image_pad = np.pad(x,((image_padding,image_padding),(image_padding,image_padding)),"constant")

    pad_0_low = image_pad.shape[0] // 2 - yh // 2
    pad_0_high = image_pad.shape[0] - yh - pad_0_low
    pad_1_low = image_pad.shape[1] // 2 - yw // 2
    pad_1_high = image_pad.shape[1] - yw - pad_1_low
    kernel_pad = np.pad(y,((pad_0_low,pad_0_high),(pad_1_low,pad_1_high)),"constant")

    kernel_shift = ifftshift(kernel_pad)
    
    print(image_pad.shape,kernel_shift.shape)
    img_fft = sdft2(image_pad)
    kernel_fft = np.fft.fft2(kernel_shift)
    conv = img_fft * kernel_fft
    print(kernel_fft)

    conv = sidft2(conv)
    return conv[image_padding:-image_padding,image_padding:-image_padding]

def simpleconv(x:np.ndarray,y:np.ndarray)->np.ndarray:
    # no fft
    yh, yw = y.shape
    xh, xw = x.shape
    assert yh == yw and xh == xw, "Only support square kernel"
    image_padding = yh//2
    image_pad = np.pad(x,((image_padding,image_padding),(image_padding,image_padding)),"constant")

    pad_0_low = image_pad.shape[0] // 2 - yh // 2
    pad_0_high = image_pad.shape[0] - yh - pad_0_low
    pad_1_low = image_pad.shape[1] // 2 - yw // 2
    pad_1_high = image_pad.shape[1] - yw - pad_1_low
    kernel_pad = np.pad(y,((pad_0_low,pad_0_high),(pad_1_low,pad_1_high)),"constant")

    conv = np.zeros_like(image_pad)
    for r in range(image_padding,image_pad.shape[0]-image_padding):
        for c in range(image_padding,image_pad.shape[1]-image_padding):
            conv[r,c] = np.sum(image_pad[r-image_padding:r+image_padding+1,c-image_padding:c+image_padding+1]*y)
    return conv[image_padding:-image_padding,image_padding:-image_padding]
    

def dft2(x:np.ndarray)->np.ndarray:

    ret = np.zeros_like(x)
    intm = np.zeros_like(x)
    rows, cols = x.shape
    for r in range(rows):
        intm[r,:] = np.array(dft(x[r]))
    for c in range(cols):
        ret[:,c] = np.array(dft(intm[:,c]))
    ret = np.roll(ret,rows//2,axis=0)
    ret = np.roll(ret,cols//2,axis=1)
    return ret

def idft2(x:np.ndarray)->np.ndarray:
    ret = np.zeros_like(x)
    intm = np.zeros_like(x)
    rows, cols = x.shape
    x = np.roll(x,-cols//2,axis=1)
    x = np.roll(x,-rows//2,axis=0)
    for c in range(cols):
        intm[:,c] = np.array(idft(x[:,c]))
    for r in range(rows):
        ret[r,:] = np.array(idft(intm[r]))
    return ret

def main1():
    from PIL import Image
    from matplotlib import pyplot as plt
    x = Image.open(r"../projects/demo/Assets/texture.png")
    x = x.resize((256-2,256-2))
    bt = x.tobytes()
    print(len(bt),x.height,x.width,x.height*x.width*3)
    w = np.frombuffer(bt,dtype=np.uint8)
    w = np.reshape(w,(x.height,x.width,3)).astype(np.complex128)
    print(w.shape)

    w = w[:,:,0] / np.max(w[:,:,0])
    fw = dft2(w)
    ifw = idft2(fw).real*255

    w = w.real
    w2 = np.stack([w,w,w],axis=-1).astype(np.float32)
    plt.imsave("w.png",w2)
    
    fw = np.sqrt(fw.real*fw.real+fw.imag*fw.imag)
    fw = np.log(fw)
    fw2 = np.stack([fw,fw,fw],axis=-1).astype(np.float32)
    fw3 = (fw2-np.min(fw2))/(np.max(fw2)-np.min(fw2))
    plt.imsave("fw.png",fw3)
    
    print(fw3)
    ifw2 = np.stack([ifw,ifw,ifw],axis=-1).astype(np.uint8)
    plt.imsave("ifw.png",ifw2)

def main2():
    x = [1,1,4,5,1,4,1,9]
    f1 = dft(x)
    f2 = ctdft(x)
    f3 = sdft(x)
    if3 = sidft(f3)
    print(f1)
    print(f2)
    print(f3)
    print(if3)

def gaussian_kernel(size:int,sigma:float)->np.ndarray:
    kernel = np.zeros((size,size))
    for r in range(size):
        for c in range(size):
            kernel[r,c] = math.exp(-((r-size//2)**2+(c-size//2)**2)/(2*sigma*sigma))
    return kernel

def sobel_kernel()->np.ndarray:
    kernel = np.zeros((3,3))
    kernel[0,:] = [-1,-2,-1]
    kernel[2,:] = [1,2,1]
    return kernel

def main3():
    from PIL import Image
    from matplotlib import pyplot as plt
    x = Image.open(r"../projects/demo/Assets/texture.png")

    kernel_size = 3
    
    x = x.resize((256-kernel_size+1,256-kernel_size+1))
    bt = x.tobytes()
    print(len(bt),x.height,x.width,x.height*x.width*3)
    w = np.frombuffer(bt,dtype=np.uint8)
    w = np.reshape(w,(x.height,x.width,3)).astype(np.complex128)
    print(w.shape)

    w = w[:,:,0] / np.max(w[:,:,0])
    kernel = sobel_kernel()#gaussian_kernel(kernel_size,55)

    res_fft = fftconv(w,kernel).real
    res_simple = simpleconv(w,kernel).real
    res_fftnp = fftconvnp(w,kernel).real

    plt.figure()
    plt.subplot(221)
    plt.imshow(w.astype(np.float32),cmap="gray")
    plt.subplot(222)
    plt.imshow(res_fft.astype(np.float32),cmap="gray")
    plt.subplot(223)
    plt.imshow(res_simple.astype(np.float32),cmap="gray")
    plt.subplot(224)
    plt.imshow(res_fftnp.astype(np.float32),cmap="gray")
    plt.show()


if __name__ == "__main__":
    main3()