from lucienii import *
from lupy import *
import math

if __name__ == '__main__':
    fin = 'SMIR.Brain.XX.O.CT_4DPWI.345575.nii'
    img = sitk.ReadImage(fin)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()

    arr = maxmin_norm(arr)
    print(np.max(arr), np.min(arr))
    print('shape: ', arr.shape)
    print('spacing: ', spacing)
    print('origin: ', origin)
    print('direction: ', direction)
    folder = 'max-min'
    if not os.path.exists(folder):
        os.makedirs(folder)

    xaif = arr.shape[2]//2
    yaif = arr.shape[3]//2
    xvof = arr.shape[2]//2
    yvof = arr.shape[3]*3//4
    win = 20
    hin = 20
    wout = 50
    hout = 20

    arr_aif = arr[:,:,xaif-win:xaif+win, yaif-hin:yaif+hin]
    arr_vof = arr[:,:,xvof-win:xvof+win, yvof-hin:yvof+hin]

    print('aif shape:', arr_aif.shape, ',vof shape:', arr_vof.shape)

    out = np.zeros((8,256,256), dtype=np.float32)
    for z in range(arr.shape[1]):
        for x in range(arr.shape[2]):
            for y in range(arr.shape[3]):
                out[z,x,y] = np.max(arr[:,z,x,y]) - np.min(arr[:,z,x,y])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_yticks(range(0,256,20))
#        ax.set_ytickslabels(range(256))
        ax.set_xticks(range(0,256,20))
#        ax.set_xtickslabels(range(256))
        im = ax.imshow(out[z], cmap=plt.cm.hot_r)
        plt.colorbar(im)
        plt.title("heatmap, max-min")
        fig.savefig(folder+'/slice '+str(z+1))
#        plt.show()

"""
    slices = arr.swapaxes(0,1)
    rangex = range(80,180,25)
    rangey = range(80,180,25)
    for i , x in enumerate(slices):
        print('############################',i+1,'###############################')
        print(x.shape)
        folder = 'slices '+str(i+1)
        if not os.path.exists(folder):
            os.makedirs(folder)
        print(x.shape)
        for j in rangex:
            for k in rangey:
                plt.plot(range(49), x[:,j,k])
        plt.show()
"""
