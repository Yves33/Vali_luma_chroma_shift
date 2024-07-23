from PIL import Image,ImageShow
import numpy as np
import PyNvCodec as nvc
import os,sys

try:
    import pycuda
    import pycuda.autoinit
    CUDAAVAIL=True
except:
    CUDAAVAIL=False
    print("Pycuda is required to run this test")

def make_img(buffer,w,h,format,merge=False):
    """
    utility function to build image from buffers. requires numpy, pillow. 
    
    Parameters
    ----------
    buffer:  Numpy array
    w, h :   Width and height of image, regardeless of buffer format
    format : Format of buffer. may be RGB, RGB_Planar, NV12, YUV420p
    merge :  Whether we should reconstitute rgb image or display the whole buffer (as grayscale image)
    """
    
    match format:
        case 'RGB':
            return Image.fromarray(buffer.reshape(h,w,3))
        case 'RGB_Planar':
            if not merge:
                return Image.fromarray(buffer.reshape(h*3,w))
            else:
                return Image.fromarray(buffer.reshape(3,h,w).transpose(2,1,0)).transpose(Image.TRANSPOSE)
        case 'NV12':
            if not merge:
                return Image.fromarray(buffer.reshape(h*3//2,w))
            else:
                y=Image.fromarray(buffer[:w*h].reshape(h,w))
                u=Image.fromarray(buffer[w*h:][::2].reshape(h//2,w//2).repeat(2, axis=0).repeat(2, axis=1))
                v=Image.fromarray(buffer[w*h+1:][::2].reshape(h//2,w//2).repeat(2, axis=0).repeat(2, axis=1))
                return Image.merge('YCbCr', (y,u,v))
        case 'YUV420p':
            if not merge:
                return Image.fromarray(buffer.reshape(h*3//2,w))
            else:
                y=Image.fromarray(buffer[:w*h].reshape(h,w))
                u=Image.fromarray(buffer[w*h:w*h*5//4].reshape(h//2,w//2).repeat(2, axis=0).repeat(2, axis=1))
                v=Image.fromarray(buffer[w*h*5//4:].reshape(h//2,w//2).repeat(2, axis=0).repeat(2, axis=1))
                return Image.merge('YCbCr', (y,u,v))

class TwoPassConverter:
    def __init__(self,fmt,w,h):
        self.to_yuv=nvc.PySurfaceConverter(fmt,nvc.PixelFormat.YUV420,gpu_id=0)
        self.to_rgb=nvc.PySurfaceConverter(nvc.PixelFormat.YUV420,nvc.PixelFormat.RGB,gpu_id=0)
        self.surf_yuv = nvc.Surface.Make(format=nvc.PixelFormat.YUV420, width=w,height=h,gpu_id=0)

    def Run(self,src,dst,cc_ctx):
        success,details=self.to_yuv.Run(src,self.surf_yuv,cc_ctx)
        if not success:
            return success, details
        success,details=self.to_rgb.Run(self.surf_yuv,dst,cc_ctx)
        return success, details

class OnePassConverter:
    def __init__(self,fmt,w,h):
        self.fmt, self.w, self.h=fmt, w, h
        self.to_rgb=nvc.PySurfaceConverter(fmt,nvc.PixelFormat.RGB,gpu_id=0)
        
    def Run(self,src,dst,cc_ctx):
        success,details=self.to_rgb.Run(src,dst,cc_ctx)
        return success, details

def check(success, info):
    if not success:
        print(sys._getframe(1).f_lineno)
        print(info)

def decode_vali(srcfile,nframes=20):
    '''
    decodes first nframes using vali.
    + download to cpu, convert using PIL and save
    + convert using VALI, download to cpu and save
    '''
    basename=srcfile.split('.')[-2]
    nvDec=nvc.PyDecoder(srcfile,{}, 0)
    rgbBuffer = np.zeros((nvDec.Width(),nvDec.Height(),3),np.uint8)
    nv12Buffer= np.zeros((nvDec.Width()*(3*nvDec.Height())//2),np.uint8)
    cc_ctx=None
    print(f"#### {nvDec.Format()} {nvDec.ColorSpace()} {nvDec.ColorRange()} ####")
    if 1: #cSpace==nvc.ColorSpace.BT_709:
        nvCvt=TwoPassConverter(nvDec.Format(),nvDec.Width(),nvDec.Height())
    else:# cSpace==nvc.ColorSpace.BT_601:
        nvCvt=OnePassConverter(nvDec.Format(),nvDec.Width(),nvDec.Height())

    nvDwn = nvc.PySurfaceDownloader(gpu_id=0)
    nvCvt=TwoPassConverter(nvDec.Format(),nvDec.Width(),nvDec.Height())
    for idx in range(nframes):
        rawSurface=nvc.Surface.Make(nvDec.Format(), nvDec.Width(),nvDec.Height(), 0)
        rgbSurface = nvc.Surface.Make(nvc.PixelFormat.RGB, nvDec.Width(),nvDec.Height(), 0)
        success, info = nvDec.DecodeSingleSurface(rawSurface)
        check(success,info)
        if 1:
            ## download nv12 plane, convert with PIL and save
            success = nvDwn.Run(rawSurface, nv12Buffer)
            check(success,info)
            make_img(nv12Buffer,nvDec.Width(),nvDec.Height(),'NV12',merge=True).save(f"./{basename}/PIL_{idx+1:02}.jpg")
        if 0 and CUDAAVAIL:
            ## same, but download using pycuda (requires GpuMem())!
            ## works correcttly!
            device_to_host=pycuda.driver.Memcpy2D()
            device_to_host.set_src_device(rawSurface.PlanePtr(0).GpuMem())
            device_to_host.set_dst_host(nv12Buffer)
            device_to_host.width_in_bytes = rawSurface.PlanePtr(0).Width()
            device_to_host.src_pitch = rawSurface.PlanePtr(0).Pitch()
            device_to_host.dst_pitch = rawSurface.PlanePtr(0).Width()
            device_to_host.src_height = rawSurface.PlanePtr(0).Height()
            device_to_host.height = rawSurface.PlanePtr(0).Height()
            device_to_host(aligned=True)
            make_img(nv12Buffer,nvDec.Width(),nvDec.Height(),'NV12',merge=True).save(f"./{basename}/PIL_{idx+1:02}.jpg")

        if 1:
            ## convert to rgb, download and display
            success, info=nvCvt.Run(rawSurface,rgbSurface,cc_ctx)
            check(success,info)
            success = nvDwn.Run(rgbSurface, rgbBuffer)
            check(success,info)
            make_img(rgbBuffer,nvDec.Width(),nvDec.Height(),'RGB',merge=True).save(f"./{basename}/VALI_{idx+1:02}.jpg")

if __name__=='__main__':
    nframes=20
    for srcfile in ["./lores.mp4","./hires.mp4"]:
        basename=srcfile.split('.')[-2]
        os.makedirs(f"./{basename}/",exist_ok=True)
        decode_vali(srcfile,nframes=nframes)